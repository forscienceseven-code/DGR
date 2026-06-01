import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional, Tuple

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224

    # Initial replanning interval
    replan_steps: int = 10

    # Adaptive replanning parameters
    adaptive: bool = True
    min_replan_steps: int = 1
    max_replan_steps: int = 50
    adapt_step: int = 5

    # Overlap score parameters
    overlap_L: int = 7
    overlap_start_j: int = 1
    overlap_dims: Tuple[int, ...] = (0, 1, 2)

    # Thresholds.
    # If None, thresholds are initialized from the first observed overlap score.
    overlap_low: Optional[float] = None
    overlap_high: Optional[float] = None
    ema_alpha: float = 0.2
    low_factor: float = 0.75
    high_factor: float = 1.25

    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_trials_per_task: int = 25

    video_out_path: str = "data/libero/videos"
    seed: int = 7


def compute_overlap_score(prev_chunk, new_chunk, prev_r, L, start_j, dims):
    """
    Compute overlap disagreement between previous chunk tail and new chunk head:

        S_j = new_chunk[j] - prev_chunk[prev_r + j]

    score = sum_{j=start_j}^{L-1} ||S_j||^2

    This uses only observable action chunks.
    """
    prev_chunk = np.asarray(prev_chunk, dtype=np.float32)
    new_chunk = np.asarray(new_chunk, dtype=np.float32)

    if prev_r + L > len(prev_chunk):
        return None

    if L > len(new_chunk):
        return None

    dims = np.asarray(dims)
    prev_chunk=np.diff(prev_chunk[:,dims],axis=0)
    new_chunk=np.diff(new_chunk[:,dims],axis=0)
    score=np.linalg.norm(prev_chunk[prev_r]-new_chunk[0])
    return score


def choose_next_replan_steps(
    current_r,
    overlap_score,
    ema_score,
    args,
):
    """
    Update replanning interval.

    Large overlap score -> plans disagree -> reduce r.
    Small overlap score -> plans agree -> increase r.
    """
    if overlap_score is None:
        return current_r, ema_score, "no_score"

    if ema_score is None:
        ema_score = overlap_score
    else:
        ema_score = args.ema_alpha * overlap_score + (1.0 - args.ema_alpha) * ema_score

    low = args.overlap_low
    high = args.overlap_high

    if low is None:
        low = args.low_factor * ema_score
    if high is None:
        high = args.high_factor * ema_score

    if overlap_score > high:
        next_r = current_r - args.adapt_step
        decision = "decrease"
    elif overlap_score < low:
        next_r = current_r + args.adapt_step
        decision = "increase"
    else:
        next_r = current_r
        decision = "keep"

    next_r = int(np.clip(next_r, args.min_replan_steps, args.max_replan_steps))
    return next_r, ema_score, decision


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            # Adaptive state
            current_r = int(np.clip(args.replan_steps, args.min_replan_steps, args.max_replan_steps))
            ema_score = None
            prev_chunk = None
            prev_r = None

            episode_overlap_scores = []
            episode_r_values = []

            logging.info(f"Starting episode {task_episodes + 1}...")

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    replay_images.append(img)

                    if not action_plan:
                        state_vec = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ).astype(np.float32)

                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state_vec,
                            "prompt": str(task_description),
                        }

                        new_chunk = np.asarray(client.infer(element)["actions"], dtype=np.float32)

                        if len(new_chunk) < args.max_replan_steps:
                            raise ValueError(
                                f"Policy predicted only {len(new_chunk)} actions, "
                                f"but max_replan_steps={args.max_replan_steps}."
                            )

                        overlap_score = None
                        decision = "initial"

                        if args.adaptive and prev_chunk is not None and prev_r is not None:
                            overlap_score = compute_overlap_score(
                                prev_chunk=prev_chunk,
                                new_chunk=new_chunk,
                                prev_r=prev_r,
                                L=args.overlap_L,
                                start_j=args.overlap_start_j,
                                dims=args.overlap_dims,
                            )

                            current_r, ema_score, decision = choose_next_replan_steps(
                                current_r=current_r,
                                overlap_score=overlap_score,
                                ema_score=ema_score,
                                args=args,
                            )

                            if overlap_score is not None:
                                episode_overlap_scores.append(overlap_score)

                            logging.info(
                                f"Adaptive replan: score={overlap_score}, "
                                f"ema={ema_score}, decision={decision}, next_r={current_r}"
                            )

                        else:
                            logging.info(f"Initial replan interval: r={current_r}")

                        current_r = int(np.clip(current_r, args.min_replan_steps, args.max_replan_steps))
                        print("using r equal to ", current_r)
                        assert len(new_chunk) >= current_r, (
                            f"We want to execute {current_r} steps, "
                            f"but policy only predicted {len(new_chunk)} steps."
                        )

                        action_plan.extend(new_chunk[:current_r])

                        prev_chunk = new_chunk
                        prev_r = current_r
                        episode_r_values.append(current_r)

                    action = action_plan.popleft()

                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break

                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")

            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{episode_idx}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

            if episode_r_values:
                logging.info(
                    f"Adaptive r stats: mean={np.mean(episode_r_values):.2f}, "
                    f"min={np.min(episode_r_values)}, max={np.max(episode_r_values)}"
                )

            if episode_overlap_scores:
                logging.info(
                    f"Overlap score stats: mean={np.mean(episode_overlap_scores):.6f}, "
                    f"min={np.min(episode_overlap_scores):.6f}, "
                    f"max={np.max(episode_overlap_scores):.6f}"
                )

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
