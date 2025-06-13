"""
Script to convert ManiSkill hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/maniskill/convert_maniskill_data_to_lerobot.py --data-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal, Union
import os
import json

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro

from mani_skill.utils import common

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()

def create_empty_dataset(
    repo_id: str,
    robot_type: str = "panda",
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:

    features = {
        "image": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (16,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["actions"],
        },
    }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def load_h5_data(data):
    """Load h5 data into memory for faster access."""
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

def load_raw_episode_data(
    ep_path: Path,
    json_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor]:
    """Load data from a ManiSkill trajectory file.
    
    Args:
        ep_path: Path to the hdf5 trajectory file
        json_path: Path to the corresponding JSON metadata file
    """
    with h5py.File(ep_path, "r") as ep:
        # Load JSON metadata
        with open(json_path, "r") as f:
            json_data = json.load(f)
        
        # Get trajectory key (should be traj_0, traj_1, etc.)
        traj_key = list(ep.keys())[0]
        trajectory = load_h5_data(ep[traj_key])
        traj_len = len(trajectory["actions"])

        # exclude the final observation as most learning workflows do not use it
        obs = common.index_dict_array(trajectory["obs"], slice(traj_len))
        # if eps_id == 0:
        #     self.obs = obs
        # else:
        #     self.obs = common.append_dict_array(self.obs, obs)

        # Load state data - ensure arrays are 2D before concatenation
        tcp_pose = trajectory["obs"]["extra"]["tcp_pose"]  # shape: (T, 7)
        qpos = trajectory["obs"]["agent"]["qpos"]  # shape: (T, 9)
        
        # Concatenate along the feature dimension (axis=1)
        state = torch.from_numpy(np.concatenate([tcp_pose, qpos], axis=1))  # shape: (T, 16)
        
        action = torch.from_numpy(trajectory["actions"])

        imgs_per_cam = {}
        for camera in ["base_camera", "hand_camera"]:
            if camera in trajectory["obs"]["sensor_data"]:
                img_data = trajectory["obs"]["sensor_data"][camera]["rgb"] # 128*128*3
                if img_data.ndim == 4:  # Uncompressed
                    imgs_per_cam[camera] = img_data
                else:  # Compressed
                    import cv2
                    imgs_array = []
                    for data in img_data:
                        imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
                    imgs_per_cam[camera] = np.array(imgs_array)

    return imgs_per_cam, state, action


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    task: str,
) -> LeRobotDataset:
    json_data = load_json(json_path)
    episodes = json_data["episodes"]

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]
        json_path = ep_path.with_suffix(".json")

        imgs_per_cam, state, action = load_raw_episode_data(ep_path, json_path)
        num_frames = state.shape[0]

        for i in range(num_frames):
            frame = {
                "state": state[i],
                "action": action[i],
            }

            # Add images if available
            for camera, img_array in imgs_per_cam.items():
                if camera == "base_camera":
                    frame["image"] = img_array[i]
                elif camera == "hand_camera":
                    frame["wrist_image"] = img_array[i]

            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    return dataset


def port_maniskill(
    data_dir: Path,
    repo_id: str = "Ruoxiang/maniskill_dataset",
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    push_to_hub: bool = True,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    # Validate input paths
    data_dir = Path(data_dir)
    if not data_dir.exists():
        if raw_repo_id is None:
            raise ValueError("raw_repo_id must be provided if data_dir does not exist")
        print(f"Downloading raw data from {raw_repo_id} to {data_dir}")
        download_raw(data_dir, repo_id=raw_repo_id)

    # Clean up existing dataset if it exists
    if (LEROBOT_HOME / repo_id).exists():
        print(f"Removing existing dataset at {LEROBOT_HOME / repo_id}")
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # Find all hdf5 files
    hdf5_files = sorted(data_dir.glob("trajectory_cpu.rgb*.h5"))  # Updated to match ManiSkill naming convention
    if not hdf5_files:
        raise ValueError(f"No trajectory h5 files found in {data_dir}")

    print(f"Found {len(hdf5_files)} trajectory files")

    # Create and populate dataset
    dataset = create_empty_dataset(
        repo_id,
        robot_type="panda",
        mode=mode,
        dataset_config=dataset_config,
    )
    
    try:
        dataset = populate_dataset(
            dataset,
            hdf5_files,
            task=task,
        )
        dataset.consolidate()

        if push_to_hub:
            print(f"Pushing dataset to hub: {repo_id}")
            dataset.push_to_hub()
            print("Dataset successfully pushed to hub")
    except Exception as e:
        print(f"Error during dataset creation: {str(e)}")
        raise


if __name__ == "__main__":
    tyro.cli(port_maniskill)
