import gc
import os
import pathlib
import sys
from datetime import datetime, timezone

sys.path.append(str(pathlib.Path(__file__).parent))

import numpy as np
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

from database.milvus_db import batch_insert_milvus, get_milvus_client
from database.sql_db import batch_insert_sqlite
from utils.config import (
    ANNOTATION_MODEL_VERSION,
    BATCH_SIZE_DB,
    BATCH_VIDEO,
    CLIP_DURATION,
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL_VERSION,
    FRAME_SAMPLING_RATE,
    NUM_PROCESSES,
    NUM_WORKERS,
)
from utils.dataset_adapter import load_video_entries
from utils.embedding import annotate, generate_video_embedding, read_video_frames_raw

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


client = get_milvus_client()


def video_start_end_generator(video_file, clip_duration):
    import cv2

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or total_frames <= 0:
        print(
            f"Warning: Invalid video properties for {video_file}: "
            f"fps={fps}, total_frames={total_frames}"
        )
        return [], []

    video_duration = total_frames / fps

    if video_duration <= clip_duration:
        return [0], [video_duration]

    segment_starts = np.arange(0, video_duration, clip_duration)
    segment_ends = segment_starts + clip_duration

    segment_ends[-1] = video_duration
    segment_starts[-1] = max(0.0, video_duration - clip_duration)

    return segment_starts.tolist(), segment_ends.tolist()


def process_single_video(video_entry, idx):
    video_file = video_entry["video_file"]
    base_metadata = video_entry.get("metadata", {})

    segment_starts, segment_ends = video_start_end_generator(video_file, CLIP_DURATION)
    if not segment_starts:
        return [], []

    segments_raw_frames = []
    for s, e in zip(segment_starts, segment_ends):
        frames = read_video_frames_raw(video_file, s, e, FRAME_SAMPLING_RATE)
        segments_raw_frames.append(frames or [])

    max_retry = 2
    annotate_results = [
        annotate(
            video_file,
            s,
            e,
            FRAME_SAMPLING_RATE,
            max_retry,
            preloaded_frames=frames,
        )
        for s, e, frames in zip(segment_starts, segment_ends, segments_raw_frames)
    ]

    embeddings = [
        generate_video_embedding(
            video_file,
            s,
            e,
            FRAME_SAMPLING_RATE,
            EMBEDDING_DIM,
            preloaded_frames=frames,
        )
        for s, e, frames in zip(segment_starts, segment_ends, segments_raw_frames)
    ]

    video_file_name = os.path.basename(video_file)
    video_file_path = os.path.dirname(video_file)
    sqlite_batch = []
    milvus_batch = []

    for s, e, emb, anno in zip(
        segment_starts, segment_ends, embeddings, annotate_results
    ):
        if emb is None:
            continue

        segment_id = (
            f"{base_metadata.get('clip_id', video_file_name)}_"
            f"{int(float(s) * 1000)}_{int(float(e) * 1000)}"
        )

        segment_metadata = {
            **base_metadata,
            "segment_id": segment_id,
            "embedding_model": EMBEDDING_MODEL_VERSION,
            "annotation_model": ANNOTATION_MODEL_VERSION,
            "processing_ts": datetime.now(timezone.utc).isoformat(),
            "clip_file_name": video_file_name,
            "clip_file_path": video_file_path,
        }

        sqlite_batch.append(
            {
                "video_file_name": video_file_name,
                "video_file_path": video_file_path,
                "segment_start": s,
                "segment_end": e,
                "annotation": anno,
                "metadata": segment_metadata,
            }
        )

        milvus_batch.append(
            {
                "video_file_name": video_file_name,
                "video_file_path": video_file_path,
                "segment_start": s,
                "segment_end": e,
                "dataset_name": base_metadata.get("dataset_name", "activitynet"),
                "camera_channel": base_metadata.get("camera_channel", "unknown"),
                "clip_id": base_metadata.get("clip_id", video_file_name),
                "scene_token": base_metadata.get("scene_token"),
                "embedding_model": EMBEDDING_MODEL_VERSION,
                "clip_vector": emb,
            }
        )
    return sqlite_batch, milvus_batch


def train():
    video_entries = load_video_entries()
    client = get_milvus_client()

    sqlite_data = []
    milvus_data = []

    with tqdm(total=len(video_entries), desc="Processing videos") as pbar:
        for i in range(0, len(video_entries), BATCH_VIDEO):
            batch_videos = video_entries[i : i + BATCH_VIDEO]
            parallel_results = Parallel(n_jobs=NUM_PROCESSES, backend="threading")(
                delayed(process_single_video)(video_entry, idx)
                for idx, video_entry in enumerate(batch_videos)
            )

            for sqlite_batch, milvus_batch in parallel_results:
                sqlite_data.extend(sqlite_batch)
                milvus_data.extend(milvus_batch)

            while len(sqlite_data) >= BATCH_SIZE_DB:
                batch_insert_sqlite(sqlite_data[:BATCH_SIZE_DB])
                batch_insert_milvus(
                    client, COLLECTION_NAME, milvus_data[:BATCH_SIZE_DB]
                )
                sqlite_data = sqlite_data[BATCH_SIZE_DB:]
                milvus_data = milvus_data[BATCH_SIZE_DB:]

            torch.cuda.empty_cache()
            gc.collect()

            pbar.update(len(batch_videos))

    if sqlite_data:
        batch_insert_sqlite(sqlite_data)
        batch_insert_milvus(client, COLLECTION_NAME, milvus_data)

    print("All videos processed & all data inserted!")


if __name__ == "__main__":
    torch.set_num_threads(NUM_WORKERS)
    train()
