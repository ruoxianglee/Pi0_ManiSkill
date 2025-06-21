import logging
import math
import pathlib
import imageio
import tyro
from termcolor import cprint
import os
import time
from typing import Optional
import argparse
import numpy as np
import dataclasses
import jax
import wandb
import torch
import collections
import tqdm
import random

from typing import Dict
from functools import partial

import gymnasium.spaces.utils
from gymnasium.vector.utils import batch_space
from mani_skill.envs.sapien_env import BaseEnv
import mani_skill.envs
import gymnasium as gym
from mani_skill.utils import gym_utils
from mani_skill.utils import common
from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack, RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

MANISKILL_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MANISKILL_ENV_RESOLUTION = 128  # resolution used to render training data

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model parameters
    #################################################################################################################
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # ManiSkill environment-specific parameters (need to be specified when launching)
    #################################################################################################################
    """the id of the environment"""
    env_id: str = "StackCube-v1"
    """the description of the task"""
    prompt: str = 'pick up a red cube and stack it on top of a green cube and let go of the cube without it falling'
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    max_episode_steps: int = 150

    #################################################################################################################
    # ManiSkill environment-specific parameters
    #################################################################################################################
    """Number of rollouts per task"""
    num_trials_per_task: int = 50
    """seed of the experiment"""
    seed: int = 0  # Random Seed (for reproducibility)
    """if toggled, cuda will be enabled by default"""
    cuda: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    torch_deterministic: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_video: bool = True
    """Path to save videos"""
    video_out_path: str = "/videos"
    """the number of parallel environments to evaluate the agent on"""
    num_eval_envs: int = 1
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    sim_backend: str = "physx_cpu"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    control_mode: str = "pd_ee_delta_pose"

class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env, rgb=True, depth=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                images.append(cam_data["rgb"])
            if self.include_depth:
                images.append(cam_data["depth"])

        ret = dict()

        if self.include_state:
            ret["state"] = np.concatenate(
                                (
                                    observation["extra"]["tcp_pose"][:,:][0], # dim: 7=3+4
                                    observation["agent"]["qpos"][:,:][0], # dim: 9=7+2
                                )
                            )
        if self.include_rgb and not self.include_depth:
            # images has shape (2, 1, 128, 128, 3)
            ret["base_img"] = images[0][0] # shape: (128, 128, 3)
            ret["wrist_img"] = images[1][0] # shape: (128, 128, 3)

        return ret

def make_eval_envs(
    env_id,
    num_envs: int,
    sim_backend: str,
    env_kwargs: dict,
    video_dir: Optional[str] = None,
    wrappers: list[gym.Wrapper] = [],
):
    """Create vectorized environment for evaluation and/or recording videos.
    For CPU vectorized environments only the first parallel environment is used to record videos.
    For GPU vectorized environments all parallel environments are used to record videos.

    Args:
        env_id: the environment id
        num_envs: the number of parallel environments
        sim_backend: the simulation backend to use. can be "cpu" or "gpu
        env_kwargs: the environment kwargs. You can also pass in max_episode_steps in env_kwargs to override the default max episode steps for the environment.
        video_dir: the directory to save the videos. If None no videos are recorded.
        wrappers: the list of wrappers to apply to the environment.
    """
    if sim_backend == "physx_cpu":

        def cpu_make_env(
            env_id, seed, video_dir=None, env_kwargs=dict()
        ):
            def thunk():
                env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
                for wrapper in wrappers:
                    env = wrapper(env)
                # env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
                env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
                if video_dir:
                    env = RecordEpisode(
                        env,
                        output_dir=video_dir,
                        save_trajectory=False,
                        info_on_video=True,
                        source_type="Pi0_ManiSkill",
                        source_desc="Pi0_ManiSkill evaluation",
                    )
                env.action_space.seed(seed)
                env.observation_space.seed(seed)
                return env

            return thunk()

        assert num_envs == 1
        seed = num_envs - 1
        env = cpu_make_env(
                    env_id,
                    seed,
                    video_dir if seed == 0 else None,
                    env_kwargs,
                )
    else:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            sim_backend=sim_backend,
            reconfiguration_freq=1,
            **env_kwargs
        )
        max_episode_steps = gym_utils.find_max_episode_steps_value(env)
        for wrapper in wrappers:
            env = wrapper(env)
        # env = FrameStack(env, num_stack=other_kwargs["obs_horizon"])
        if video_dir:
            env = RecordEpisode(
                env,
                output_dir=video_dir,
                save_trajectory=False,
                save_video=True,
                source_type="Pi0_ManiSkill",
                source_desc="Pi0_ManiSkill evaluation",
                max_steps_per_video=max_episode_steps,
            )
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
    return env

def eval_maniskill(args: Args) -> None:
    #################################################################################################################
    # ManiSkill environment
    #################################################################################################################
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="rgb",
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps

    # Initialize ManiSkill environment and task description
    task_description = args.prompt
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        video_dir=args.video_out_path if args.capture_video else None,
        wrappers=[partial(FlattenRGBDObservationWrapper,
                              depth=False,
                              )],
    )
    print(f"Task environment: {args.env_id}")

    #################################################################################################################
    # Pi0 Policy
    #################################################################################################################
    config = _config.get_config("pi0_maniskill")

    current_root = os.getcwd()
    checkpoint_dir = os.path.join(current_root, "checkpoints/pi0_maniskill/pi0_maniskill_stackcube")

    # Create a trained policy.
    policy = _policy_config.create_trained_policy(config, checkpoint_dir, default_prompt=args.prompt)

    #################################################################################################################
    # Start evaluation
    #################################################################################################################
    task_rollouts, task_successes = 0, 0
    for rollout_index in tqdm.tqdm(range(args.num_trials_per_task)):
        print(f"\nTask: {task_description}")

        # Reset environment
        obs, _ = envs.reset(seed=args.seed)
        action_plan = collections.deque()

        # Setup
        t = 0
        replay_images = []

        print(f"Running rollout: {task_rollouts+1}...")
        done = False
        while t < args.max_episode_steps:
            try:
                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["base_img"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["wrist_img"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                # Save preprocessed image for replay video
                replay_images.append(img)
                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict
                    element = {
                        "image": img, # shape: (224, 224, 3)
                        "wrist_image": wrist_img, # shape: (224, 224, 3)
                        "state": obs["state"], # shape: (16,)
                        "prompt": str(task_description),
                    }

                    # Query model to get action
                    action_chunk = policy.infer(element)["actions"] # 形状 (horizon, action_dim) = (10, 7)
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[: args.replan_steps])

                # Execute action in environment
                action = action_plan.popleft()
                obs, reward, terminated, truncated, info = envs.step(action.tolist())
                done = terminated
                if done:
                    task_successes += 1
                    break
                t += 1

            except Exception as e:
                logging.error(f"Caught exception: {e}")
                break

        print(f"Rollout {task_rollouts+1} completed with episodes {t}.")
        task_rollouts += 1

        # Log current results
        print(f"Success: {done}")
        print(f"# episodes completed so far: {task_rollouts}")
        print(f"# successes: {task_successes} ({task_successes / task_rollouts * 100:.1f}%)")

    # Log final results
    print(f"Current task success rate: {float(task_successes) / float(task_rollouts)}")
    print(f"Total episodes: {task_rollouts}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_maniskill)
