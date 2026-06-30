#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2


def _load_table(meta_root: Path, name: str):
    table_file = meta_root / f"{name}.json"
    with table_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get_image_size(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return image.shape[1], image.shape[0]


def _summarize_ego_motion(frame_sample_data: List[Dict], ego_pose_by_token: Dict):
    speeds = []
    for idx in range(1, len(frame_sample_data)):
        prev_sd = frame_sample_data[idx - 1]
        curr_sd = frame_sample_data[idx]

        prev_pose = ego_pose_by_token.get(prev_sd.get("ego_pose_token"))
        curr_pose = ego_pose_by_token.get(curr_sd.get("ego_pose_token"))
        if not prev_pose or not curr_pose:
            continue

        prev_t = prev_pose.get("translation", [0, 0, 0])
        curr_t = curr_pose.get("translation", [0, 0, 0])
        dx = curr_t[0] - prev_t[0]
        dy = curr_t[1] - prev_t[1]
        dz = curr_t[2] - prev_t[2]
        distance = (dx * dx + dy * dy + dz * dz) ** 0.5

        dt = (curr_sd.get("timestamp", 0) - prev_sd.get("timestamp", 0)) / 1e6
        if dt > 0:
            speeds.append(distance / dt)

    if not speeds:
        return {"ego_avg_speed_mps": None, "ego_max_speed_mps": None}

    return {
        "ego_avg_speed_mps": sum(speeds) / len(speeds),
        "ego_max_speed_mps": max(speeds),
    }


def _iter_scene_front_frames(scene: Dict, sample_by_token: Dict, sample_data_by_token: Dict):
    cursor = scene.get("first_sample_token", "")
    while cursor:
        sample = sample_by_token.get(cursor)
        if not sample:
            break
        cam_front_token = sample.get("data", {}).get("CAM_FRONT")
        if cam_front_token:
            sd = sample_data_by_token.get(cam_front_token)
            if sd:
                yield sample, sd
        cursor = sample.get("next", "")


def generate_nuscenes_front_videos(
    dataroot: str,
    meta_root: str,
    output_dir: str,
    dataset_version: str,
    dataset_split: str,
    fps: float,
    scene_tokens: Optional[List[str]] = None,
):
    dataroot_path = Path(dataroot)
    meta_root_path = Path(meta_root)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    scenes = _load_table(meta_root_path, "scene")
    samples = _load_table(meta_root_path, "sample")
    sample_data = _load_table(meta_root_path, "sample_data")
    logs = _load_table(meta_root_path, "log")

    try:
        ego_pose = _load_table(meta_root_path, "ego_pose")
    except FileNotFoundError:
        ego_pose = []

    sample_by_token = {x["token"]: x for x in samples}
    sample_data_by_token = {x["token"]: x for x in sample_data}
    log_by_token = {x["token"]: x for x in logs}
    ego_pose_by_token = {x["token"]: x for x in ego_pose}

    selected_scenes = scenes
    if scene_tokens:
        scene_token_set = set(scene_tokens)
        selected_scenes = [s for s in scenes if s.get("token") in scene_token_set]

    mapping = []
    for scene in selected_scenes:
        scene_token = scene.get("token")
        scene_name = scene.get("name", scene_token)
        log_token = scene.get("log_token")
        log_rec = log_by_token.get(log_token, {})
        location = log_rec.get("location")
        map_token = log_rec.get("map_token")

        front_frames = list(
            _iter_scene_front_frames(scene, sample_by_token, sample_data_by_token)
        )
        if not front_frames:
            continue

        first_sample, first_sd = front_frames[0]
        last_sample, last_sd = front_frames[-1]

        first_img_path = dataroot_path / first_sd["filename"]
        size = _safe_get_image_size(first_img_path)
        if not size:
            continue

        width, height = size
        clip_id = f"{scene_name}_{scene_token[:8]}_CAM_FRONT"
        clip_file_name = f"{clip_id}.mp4"
        clip_path = output_dir_path / clip_file_name

        writer = cv2.VideoWriter(
            str(clip_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        written = 0
        frame_sample_data = []
        for sample, sd in front_frames:
            img_path = dataroot_path / sd["filename"]
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
            frame_sample_data.append(sd)

        writer.release()

        if written == 0:
            if clip_path.exists():
                clip_path.unlink()
            continue

        ego_motion = _summarize_ego_motion(frame_sample_data, ego_pose_by_token)
        mapping.append(
            {
                "dataset_name": "nuscenes",
                "dataset_version": dataset_version,
                "dataset_split": dataset_split,
                "camera_channel": "CAM_FRONT",
                "clip_id": clip_id,
                "clip_file_name": clip_file_name,
                "clip_relative_path": clip_file_name,
                "scene_token": scene_token,
                "sample_start_token": first_sample.get("token"),
                "sample_end_token": last_sample.get("token"),
                "log_token": log_token,
                "location": location,
                "map_token": map_token,
                "clip_start_ts": first_sd.get("timestamp"),
                "clip_end_ts": last_sd.get("timestamp"),
                "fps": fps,
                "frame_count": written,
                "quality_flag": "ok",
                **ego_motion,
            }
        )

    mapping_path = output_dir_path / "clip_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(mapping)} CAM_FRONT clips")
    print(f"Clip mapping written to: {mapping_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic nuScenes CAM_FRONT videos from image frames"
    )
    parser.add_argument("--dataroot", required=True, help="nuScenes dataroot path")
    parser.add_argument(
        "--meta-root",
        required=True,
        help="Path to nuScenes metadata JSON directory (scene/sample/sample_data/log)",
    )
    parser.add_argument("--output-dir", required=True, help="Output clip directory")
    parser.add_argument(
        "--dataset-version", default="v1.0-trainval", help="Dataset version label"
    )
    parser.add_argument("--dataset-split", default="train", help="Dataset split label")
    parser.add_argument("--fps", type=float, default=12.0, help="Output video FPS")
    parser.add_argument(
        "--scene-token",
        action="append",
        default=[],
        help="Optional scene token filter (can be repeated)",
    )
    args = parser.parse_args()

    generate_nuscenes_front_videos(
        dataroot=args.dataroot,
        meta_root=args.meta_root,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
        dataset_split=args.dataset_split,
        fps=args.fps,
        scene_tokens=args.scene_token,
    )


if __name__ == "__main__":
    main()
