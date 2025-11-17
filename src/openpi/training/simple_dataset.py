"""Simple dataset loader for local LeRobot datasets that bypasses HuggingFace."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import SupportsIndex
import av
from PIL import Image


class SimpleLeRobotDataset:
    """Simple dataset loader that reads parquet files directly."""

    def __init__(self, root: str | Path, delta_timestamps: dict = None):
        self.root = Path(root)

        # Load metadata
        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)

        # Load all parquet files
        self.data = []
        data_dir = self.root / "data"
        for chunk_dir in sorted(data_dir.glob("chunk-*")):
            for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
                df = pd.read_parquet(parquet_file)
                self.data.append(df)

        self.data = pd.concat(self.data, ignore_index=True)
        self.fps = self.info["fps"]
        self.delta_timestamps = delta_timestamps or {}

        # Preload all video frames into memory
        print("Loading all video frames into memory...")
        self._video_frames = self._preload_all_videos()
        print(f"Loaded {len(self._video_frames)} video frames")

        # Precompute indices for action sequences
        self._prepare_action_sequences()

    def _preload_all_videos(self) -> dict:
        """Preload all video frames into memory."""
        video_frames = {}

        # Get all video keys from info
        video_keys = []
        for feature_name, feature_info in self.info["features"].items():
            if feature_info.get("dtype") == "video":
                video_keys.append(feature_name)

        # Load all frames for each video key
        for video_key in video_keys:
            video_dir = self.root / "videos" / video_key
            for episode_idx in self.data["episode_index"].unique():
                chunk_idx = episode_idx // self.info["chunks_size"]
                file_idx = (episode_idx % self.info["chunks_size"]) // 1000

                video_path = video_dir / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"

                if not video_path.exists():
                    continue

                # Load all frames for this episode
                container = av.open(str(video_path))
                frames = []
                for frame in container.decode(video=0):
                    frames.append(frame.to_ndarray(format='rgb24'))
                container.close()

                # Store frames indexed by (video_key, episode_idx)
                video_frames[(video_key, episode_idx)] = frames

        return video_frames

    def _prepare_action_sequences(self):
        """Prepare indices for loading action sequences."""
        self.valid_indices = []

        # Group by episode
        for episode_idx in self.data["episode_index"].unique():
            episode_data = self.data[self.data["episode_index"] == episode_idx]
            episode_indices = episode_data.index.tolist()

            # For each timestep in the episode, check if we can get full action sequence
            max_horizon = 0
            for key, timestamps in self.delta_timestamps.items():
                max_horizon = max(max_horizon, len(timestamps))

            for i in range(len(episode_indices) - max_horizon + 1):
                self.valid_indices.append(episode_indices[i])

    def _load_video_frame(self, video_key: str, episode_idx: int, frame_idx: int) -> np.ndarray:
        """Load a single frame from preloaded video frames."""
        frames = self._video_frames.get((video_key, episode_idx))

        if frames is None:
            raise ValueError(f"No frames found for {video_key}, episode {episode_idx}")

        # Get the frame at the correct index within the episode
        episode_data = self.data[self.data["episode_index"] == episode_idx]
        local_frame_idx = (episode_data["frame_index"] == frame_idx).idxmax()
        local_idx = episode_data.index.get_loc(local_frame_idx)

        if local_idx < len(frames):
            return frames[local_idx]
        else:
            return frames[-1]

    def __len__(self):
        return len(self.valid_indices) if self.valid_indices else len(self.data)

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()

        if self.valid_indices:
            data_idx = self.valid_indices[idx]
        else:
            data_idx = idx

        row = self.data.iloc[data_idx]
        episode_idx = row["episode_index"]
        frame_idx = row["frame_index"]

        # Load images from videos
        scene_image = self._load_video_frame("observation.images.scene", episode_idx, frame_idx)
        gripper_image = self._load_video_frame("observation.images.gripper", episode_idx, frame_idx)

        # Get current observation
        result = {
            "observation.state": np.array(row["observation.state"], dtype=np.float32),
            "observation.images.scene": scene_image,
            "observation.images.gripper": gripper_image,
            "episode_index": episode_idx,
            "frame_index": frame_idx,
        }

        # Get action sequence if delta_timestamps specified
        if self.delta_timestamps:
            # Use the first key from delta_timestamps, but store as "action"
            timestamps = list(self.delta_timestamps.values())[0]
            actions = []
            episode_data = self.data[self.data["episode_index"] == episode_idx]

            for dt in timestamps:
                target_frame = frame_idx + int(dt * self.fps)
                # Find the action at target frame
                target_row = episode_data[episode_data["frame_index"] == target_frame]
                if len(target_row) > 0:
                    actions.append(target_row.iloc[0]["action"])
                else:
                    # If frame doesn't exist, use last action
                    actions.append(episode_data.iloc[-1]["action"])

            result["action"] = np.array(actions, dtype=np.float32)
        else:
            result["action"] = np.array([row["action"]], dtype=np.float32)

        # Add task if available
        if "task_index" in row:
            result["task_index"] = row["task_index"]

        return result


class SimpleLeRobotDatasetMetadata:
    """Simple metadata loader."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

        with open(self.root / "meta" / "info.json") as f:
            self.info = json.load(f)

        self.fps = self.info["fps"]
        self.features = self.info["features"]

        # Load tasks
        self.tasks = {}
        # Try tasks.parquet first (newer format)
        tasks_parquet = self.root / "meta" / "tasks.parquet"
        if tasks_parquet.exists():
            tasks_df = pd.read_parquet(tasks_parquet)
            # The parquet format has task strings as index and task_index as column
            for task_str, row in tasks_df.iterrows():
                task_index = int(row["task_index"])
                self.tasks[task_index] = task_str
        else:
            # Fall back to tasks.jsonl (older format)
            tasks_file = self.root / "meta" / "tasks.jsonl"
            if tasks_file.exists():
                with open(tasks_file) as f:
                    for line in f:
                        task = json.loads(line)
                        self.tasks[task["task_index"]] = task.get("task", "")
