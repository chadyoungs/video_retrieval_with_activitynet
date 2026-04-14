import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent))

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from joblib import Parallel, delayed
from pymilvus import MilvusClient

from database.milvus_db import batch_insert_milvus
from database.sql_db import batch_save_to_db
from utils.config import (
    ALIAS,
    BATCH_SIZE_DB,
    CLIP_DURATION,
    COLLECTION_NAME,
    FRAME_SAMPLING_RATE,
    MILVUS_HOST,
    MILVUS_PORT,
    NUM_PROCESSES,
    NUM_WORKERS,
)
from utils.embedding import EMBEDDING_DIM, annotate, generate_video_embedding

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_ANNOTATION = True


_client = None


def get_milvus_client():
    global _client
    if _client is None:
        try:
            _client = MilvusClient(
                uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", alias=ALIAS
            )
            print(f"Connected to Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
        except Exception as e:
            raise RuntimeError(f"Milvus connection failed: {e}")
    return _client


def get_video_file_list(data_root, file_format="avi"):
    return [str(p) for p in Path(data_root).glob(f"*.{file_format}")]


def video_start_end_generator(video_file, clip_duration):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = total_frames / fps
    cap.release()

    if video_duration <= clip_duration:
        return [0], [video_duration]

    segment_starts = np.arange(0, video_duration, clip_duration)
    segment_ends = segment_starts + clip_duration

    segment_ends[-1] = video_duration
    segment_starts[-1] = video_duration - clip_duration

    return segment_starts.tolist(), segment_ends.tolist()


@lru_cache(maxsize=10000)
def cached_annotate(video_file, segment_start, segment_end, frame_sampling_rate):
    return annotate(
        video_file, segment_start, segment_end, frame_sampling_rate, max_retries=1
    )


@lru_cache(maxsize=10000)
def cached_generate_video_embedding(
    video_file, segment_start, segment_end, sample_rate
):
    return generate_video_embedding(
        video_file, segment_start, segment_end, sample_rate, EMBEDDING_DIM
    )


def process_single_video(video_file, idx):
    segment_starts, segment_ends = video_start_end_generator(video_file, CLIP_DURATION)
    video_segments = [
        (video_file, s, e, FRAME_SAMPLING_RATE)
        for s, e in zip(segment_starts, segment_ends)
    ]
    if not video_segments:
        return [], []

    if CACHE_ANNOTATION:
        annotate_results = [cached_annotate(*seg) for seg in video_segments]
        embeddings = [cached_generate_video_embedding(*seg) for seg in video_segments]
    else:
        raise NotImplementedError("Non-cached annotation is not implemented yet")

    video_file_name = os.path.basename(video_file)
    video_file_path = os.path.dirname(video_file)
    sqlite_batch = []
    milvus_batch = []
    for i, (s, e, emb, anno) in enumerate(
        zip(segment_starts, segment_ends, embeddings, annotate_results)
    ):
        if emb is None:
            continue
        # SQLite batch
        sqlite_batch.append((video_file_name, video_file_path, s, e, anno))
        # Milvus batch
        milvus_batch.append(
            {
                "video_id": idx,
                "video_file_name": video_file_name,
                "video_file_path": video_file_path,
                "segment_start": s,
                "segment_end": e,
                "clip_vector": emb,
            }
        )
    return sqlite_batch, milvus_batch


def train():
    data_root = "/mnt/sdc/activitynet_caption/demo_data"
    file_format = "mp4"
    video_file_list = get_video_file_list(data_root, file_format)
    client = get_milvus_client()

    parallel_results = Parallel(n_jobs=NUM_PROCESSES, backend="threading")(
        delayed(process_single_video)(video_file, idx)
        for idx, video_file in enumerate(video_file_list)
    )

    all_sqlite_data = []
    all_milvus_data = []
    for sqlite_batch, milvus_batch in parallel_results:
        all_sqlite_data.extend(sqlite_batch)
        all_milvus_data.extend(milvus_batch)

    for i in range(0, len(all_sqlite_data), BATCH_SIZE_DB):
        batch = all_sqlite_data[i : i + BATCH_SIZE_DB]
        insert_result_sqlite = batch_save_to_db(batch)
        print(f"SQLite batch {i//BATCH_SIZE_DB} inserted: {len(batch)} rows")

    for i in range(0, len(all_milvus_data), BATCH_SIZE_DB):
        batch = all_milvus_data[i : i + BATCH_SIZE_DB]
        insert_result_milvus = batch_insert_milvus(client, COLLECTION_NAME, batch)
        print(
            f"Milvus batch {i//BATCH_SIZE_DB} inserted: {len(batch)} rows, ID: {insert_result_milvus}"
        )


if __name__ == "__main__":
    torch.set_num_threads(NUM_WORKERS)
    train()
