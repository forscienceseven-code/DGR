import collections
import os
import pprint
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import tqdm
import tyro
from libero.libero import benchmark

from examples.Libero.eval.utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    normalize_gripper_action,
    quat2axisangle,
    save_rollout_video,
)

log_dir = "/tmp/logs"
os.makedirs(log_dir, exist_ok=True)


def summarize_obs(obs_dict):
    summary = {}
    for k, v in obs_dict.items():
        if isinstance(v, torch.Tensor):
            summary[k] = {"shape": tuple(v.shape), "dtype": v.dtype, "device": v.device}
        elif isinstance(v, np.ndarray):
            summary[k] = {"shape": v.shape, "dtype": v.dtype}
        else:
            summary[k] = type(v).__name__
    pprint.pprint(summary)


def show_obs_images_cv2(new_obs):
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]

    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)

    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


@dataclass
class GenerateConfig:
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 5

    port: int = 5555
    headless: bool = False

    # Initial replanning interval
    replan_steps: int = 5

    # Adaptive replanning parameters
    adaptive: bool = True
    min_replan_steps: int = 1
    max_replan_steps: int = 16
    adapt_step: int = 2

    # Overlap score parameters
    overlap_dims: Tuple[int, ...] = (0, 1, 2)

    # Thresholds
    overlap_low: Optional[float] = None
    overlap_high: Optional[float] = None
    ema_alpha: float = 0.2
    low_factor: float = 0.75
    high_factor: float = 1.25


class GR00TPolicy:
    """GR00T Policy wrapper for LIBERO with adaptive action-chunk execution."""

    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(
        self,
        host="localhost",
        port=5555,
        headless=False,
        replan_steps=8,
        adaptive=True,
        min_replan_steps=1,
        max_replan_steps=16,
        adapt_step=3,
        overlap_dims=(0, 1, 2),
        overlap_low=None,
        overlap_high=None,
        ema_alpha=0.2,
        low_factor=0.75,
        high_factor=1.25,
    ):
        from gr00t.eval.service import ExternalRobotInferenceClient

        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.config = self.LIBERO_CONFIG
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        self.headless = headless

        self.action_plan = collections.deque()

        self.adaptive = adaptive
        self.current_r = int(np.clip(replan_steps, min_replan_steps, max_replan_steps))
        self.min_replan_steps = min_replan_steps
        self.max_replan_steps = max_replan_steps
        self.adapt_step = adapt_step
        self.overlap_dims = tuple(overlap_dims)

        self.overlap_low = overlap_low
        self.overlap_high = overlap_high
        self.ema_alpha = ema_alpha
        self.low_factor = low_factor
        self.high_factor = high_factor

        self.initial_replan_steps = int(np.clip(replan_steps, min_replan_steps, max_replan_steps))
        self.prev_chunk = None
        self.prev_r = None
        self.ema_score = None

    def reset(self):
        self.action_plan.clear()
        self.prev_chunk = None
        self.prev_r = None
        self.current_r = self.initial_replan_steps
        self.ema_score = None

    def _max_comparable_replan_steps(self):
        # For a chunk with T actions there are T - 1 action deltas.
        # We compare prev_delta[prev_r] with new_delta[0], so prev_r must be <= T - 2.
        return self.max_replan_steps - 2

    def get_action(self, observation_dict, lang: str,number=0,numbers=[]):
        """Return one LIBERO action.

        Internally:
            observe -> get GR00T chunk -> choose r -> execute first r actions one by one
        """
        if not self.action_plan:
            number=number+1
            obs_dict = self._process_observation(observation_dict, lang)
            action_chunk_dict = self.policy.get_action(obs_dict)
            new_chunk = self._action_chunk_dict_to_array(action_chunk_dict)
            #print(new_chunk.shape)

            if len(new_chunk) < self.max_replan_steps:
                raise ValueError(
                    f"GR00T predicted only {len(new_chunk)} actions, "
                    f"but max_replan_steps={self.max_replan_steps}."
                )

            overlap_score = None
            decision = "initial"

            if self.adaptive and self.prev_chunk is not None and self.prev_r is not None:
                max_comparable_r = self._max_comparable_replan_steps()

                if self.prev_r > max_comparable_r:
                    # The previous execution reached too far into the chunk to form a valid
                    # delta-overlap with the newly sampled chunk. Reset to the largest
                    # comparable r instead of forcing an invalid/hardcoded index.
                    decision = "no_overlap_after_full_horizon"
                    self.current_r = max_comparable_r
                    print(
                        f"Adaptive replan: prev_r={self.prev_r} has no valid diff-overlap. "
                        f"Resetting next_r={self.current_r}"
                    )
                else:
                    overlap_score = self._compute_overlap_score(
                        prev_chunk=self.prev_chunk,
                        new_chunk=new_chunk,
                        prev_r=self.prev_r,
                    )

                    if overlap_score is None:
                        decision = "no_score"
                        self.current_r = max_comparable_r
                    else:
                        self.current_r, self.ema_score, decision = self._choose_next_replan_steps(
                            current_r=self.current_r,
                            overlap_score=overlap_score,
                            ema_score=self.ema_score,
                        )

                    print(
                        f"Adaptive replan: score={overlap_score}, "
                        f"ema={self.ema_score}, "
                        f"decision={decision}, "
                        f"next_r={self.current_r}"
                    )
            else:
                print(f"Initial replan interval: r={self.current_r}")

            self.current_r = int(
                np.clip(self.current_r, self.min_replan_steps, self.max_replan_steps)
            )

            assert len(new_chunk) >= self.current_r, (
                f"We want to execute {self.current_r} steps, "
                f"but GR00T only predicted {len(new_chunk)} steps."
            )

            self.action_plan.extend(new_chunk[: self.current_r])

            self.prev_chunk = new_chunk
            self.prev_r = self.current_r

        action = self.action_plan.popleft()
        action = normalize_gripper_action(action, binarize=True)

        assert len(action) == 7, f"Expected 7-dim action, got {len(action)}"
        return action,number

    def _action_chunk_dict_to_array(self, action_chunk: dict[str, np.ndarray]) -> np.ndarray:
        """Convert GR00T action dict into shape [T, 7]."""
        components = []

        for key in self.action_keys:
            arr = np.asarray(action_chunk[f"action.{key}"], dtype=np.float32)
            arr = np.squeeze(arr)

            if arr.ndim == 0:
                arr = arr[None]

            components.append(arr)

        min_len = min(len(x) for x in components)
        components = [x[:min_len] for x in components]

        return np.stack(components, axis=-1).astype(np.float32)

    def _compute_overlap_score(self, prev_chunk, new_chunk, prev_r):
        """Compare the first available diff-overlap: prev_delta[prev_r] vs new_delta[0]."""
        prev_chunk = np.asarray(prev_chunk, dtype=np.float32)
        new_chunk = np.asarray(new_chunk, dtype=np.float32)

        dims = np.asarray(self.overlap_dims)

        prev_delta = np.diff(prev_chunk[:, dims], axis=0)
        new_delta = np.diff(new_chunk[:, dims], axis=0)

        prev_idx = prev_r
        new_idx = 0

        if prev_idx >= len(prev_delta) or new_idx >= len(new_delta):
            return None

        diff = prev_delta[prev_idx] - new_delta[new_idx]
        return float(np.sum(diff ** 2))

    def _choose_next_replan_steps(self, current_r, overlap_score, ema_score):
        if overlap_score is None:
            return current_r, ema_score, "no_score"

        if ema_score is None:
            ema_score = overlap_score
        else:
            ema_score = self.ema_alpha * overlap_score + (1.0 - self.ema_alpha) * ema_score

        low = self.overlap_low
        high = self.overlap_high

        if low is None:
            low = self.low_factor * ema_score
        if high is None:
            high = self.high_factor * ema_score

        if overlap_score > high:
            next_r = current_r - self.adapt_step
            decision = "decrease"
        elif overlap_score < low:
            next_r = current_r + self.adapt_step
            decision = "increase"
        else:
            next_r = current_r
            decision = "keep"

        next_r = int(np.clip(next_r, self.min_replan_steps, self.max_replan_steps))
        return next_r, ema_score, decision

    def _process_observation(self, obs, lang: str):
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)

        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }

        if not self.headless:
            show_obs_images_cv2(new_obs)

        return new_obs


def eval_libero(cfg: GenerateConfig) -> None:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks

    print(f"Task suite: {cfg.task_suite_name}")

    log_file = open(f"{log_dir}/libero_eval_{cfg.task_suite_name}.log", "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    total_episodes, total_successes = 0, 0
    numbers=[]
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        env, task_description = get_libero_env(task, resolution=256)

        gr00t_policy = GR00TPolicy(
            host="localhost",
            port=cfg.port,
            headless=cfg.headless,
            replan_steps=cfg.replan_steps,
            adaptive=True,
            min_replan_steps=cfg.min_replan_steps,
            max_replan_steps=cfg.max_replan_steps,
            adapt_step=cfg.adapt_step,
            overlap_dims=cfg.overlap_dims,
            overlap_low=cfg.overlap_low,
            overlap_high=cfg.overlap_high,
            ema_alpha=cfg.ema_alpha,
            low_factor=cfg.low_factor,
            high_factor=cfg.high_factor,
        )

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            env.reset()
            gr00t_policy.reset()

            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            top_view = []
            wrist_view = []
            done = False

            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 600
            elif cfg.task_suite_name == "libero_10":
                max_steps = 1200
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400
            else:
                raise ValueError(f"Unknown task suite: {cfg.task_suite_name}")

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")
            number=0
            while t < max_steps + cfg.num_steps_wait:
                if t>max_steps-10:
                 print("time up!")
                try:
                    if done:
                       break
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    img, wrist_img = get_libero_image(obs)

                    top_view.append(img)
                    wrist_view.append(wrist_img)

                    action,number = gr00t_policy.get_action(obs, task.language,number=number,numbers=numbers)
                    #print(f"STEP {t}, done before step = {done}")
                    obs, reward, done, info = env.step(action.tolist())
                    #print(f"STEP {t}, done after step = {done}, reward = {reward}, info = {info}")

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            numbers.append(number)
            save_rollout_video(
                top_view,
                wrist_view,
                total_episodes,
                success=done,
                task_description=task_description,
                log_file=log_file,
            )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(
                f"# successes: {total_successes} "
                f"({total_successes / total_episodes * 100:.1f}%)\n"
            )
            log_file.flush()

        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

        log_file.write(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}\n"
        )
        log_file.write(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        avg=np.mean(numbers)
        stdErr=np.std(numbers)
        print("avg and std number of calls ",avg,stdErr)
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    cfg = tyro.cli(GenerateConfig)
    eval_libero(cfg)
