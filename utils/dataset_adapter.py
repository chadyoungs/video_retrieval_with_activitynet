import json
from pathlib import Path
from typing import Dict, List

from utils.config import (
    ACTIVITYNET_DATA_ROOT,
    ACTIVITYNET_FILE_FORMAT,
    DATASET_SOURCE,
    NUSCENES_CLIP_MAPPING_FILE,
    NUSCENES_DATASET_NAME,
    NUSCENES_DATASET_SPLIT,
    NUSCENES_DATASET_VERSION,
    NUSCENES_VIDEO_ROOT,
)


ALLOWED_DATASET_SOURCES = {"activitynet", "nuscenes"}


def _build_activitynet_entries() -> List[Dict]:
    data_root = Path(ACTIVITYNET_DATA_ROOT)
    return [
        {
            "video_file": str(p),
            "metadata": {
                "dataset_name": "activitynet",
                "dataset_version": "unknown",
                "dataset_split": "unknown",
                "camera_channel": "unknown",
            },
        }
        for p in data_root.glob(f"*.{ACTIVITYNET_FILE_FORMAT}")
    ]


def _build_nuscenes_entries() -> List[Dict]:
    mapping_file = Path(NUSCENES_CLIP_MAPPING_FILE)
    if not mapping_file.exists():
        raise FileNotFoundError(
            f"nuScenes clip mapping file not found: {NUSCENES_CLIP_MAPPING_FILE}"
        )

    with mapping_file.open("r", encoding="utf-8") as f:
        clip_mapping = json.load(f)

    video_root = Path(NUSCENES_VIDEO_ROOT)
    entries = []
    for item in clip_mapping:
        clip_rel_path = item.get("clip_relative_path") or item.get("clip_file_name")
        if not clip_rel_path:
            continue

        clip_path = video_root / clip_rel_path
        if not clip_path.exists():
            continue

        metadata = {
            "dataset_name": NUSCENES_DATASET_NAME,
            "dataset_version": item.get("dataset_version", NUSCENES_DATASET_VERSION),
            "dataset_split": item.get("dataset_split", NUSCENES_DATASET_SPLIT),
            "camera_channel": item.get("camera_channel", "CAM_FRONT"),
            "clip_id": item.get("clip_id"),
            "scene_token": item.get("scene_token"),
            "sample_start_token": item.get("sample_start_token"),
            "sample_end_token": item.get("sample_end_token"),
            "log_token": item.get("log_token"),
            "location": item.get("location"),
            "map_token": item.get("map_token"),
            "clip_start_ts": item.get("clip_start_ts"),
            "clip_end_ts": item.get("clip_end_ts"),
            "clip_fps": item.get("fps"),
            "clip_frame_count": item.get("frame_count"),
            "ego_avg_speed_mps": item.get("ego_avg_speed_mps"),
            "ego_max_speed_mps": item.get("ego_max_speed_mps"),
            "quality_flag": item.get("quality_flag", "ok"),
        }
        entries.append({"video_file": str(clip_path), "metadata": metadata})

    return entries


def load_video_entries() -> List[Dict]:
    if DATASET_SOURCE not in ALLOWED_DATASET_SOURCES:
        raise ValueError(
            f"Unsupported DATASET_SOURCE={DATASET_SOURCE}. "
            f"Expected one of {sorted(ALLOWED_DATASET_SOURCES)}"
        )

    if DATASET_SOURCE == "nuscenes":
        return _build_nuscenes_entries()
    return _build_activitynet_entries()
