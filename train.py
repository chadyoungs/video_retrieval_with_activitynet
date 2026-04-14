import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import cv2
import torch  # PyTorch is needed for CLIP model inference
from pymilvus import MilvusClient, Collection, connections

from database.sql_db import save_to_db
from utils.embedding import generate_video_embedding, annotate, EMBEDDING_DIM
from utils.config import CLIP_DURATION, FRAME_SAMPLING_RATE, COLLECTION_NAME, ALIAS, MILVUS_HOST, MILVUS_PORT


# Establish the connection
try:
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", alias=ALIAS)
    print(
        f"Successfully connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT} using alias '{ALIAS}'."
    )

except Exception as e:
    # Handle the error, perhaps exit the script
    print(f"Failed to connect to Milvus: {e}")


def get_video_file_list(data_root, file_format="avi"):
    file_list = [
        os.path.join(data_root, file) for file in os.listdir(data_root) if file.endswith(file_format)
    ]

    return file_list

def video_start_end_generator(video_file, clip_duration):
    """
    Generate start and end times for video segmentation based on clip duration
    """
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_duration = total_frames / fps

    segment_starts = []
    segment_ends = []

    current_time = 0.0
    while current_time < video_duration:
        segment_starts.append(current_time)
        segment_ends.append(min(current_time + clip_duration, video_duration))
        current_time += clip_duration

    cap.release()
    
    segment_starts[-1] = video_duration - clip_duration # Ensure the last segment is exactly clip_duration long
    
    return segment_starts, segment_ends

def train():
    data_root = "/mnt/sdc/activitynet_caption/demo_data"
    file_format = "mp4"
    video_file_list = get_video_file_list(data_root, file_format)

    # Process the video dataset
    for idx, video_file in enumerate(video_file_list):
        segment_starts, segment_ends = video_start_end_generator(video_file, CLIP_DURATION)
        for segment_start, segment_end in zip(segment_starts, segment_ends):
            video_embedding = generate_video_embedding(
                video_file, segment_start, segment_end, FRAME_SAMPLING_RATE, EMBEDDING_DIM
            )
            annotate_res = annotate(video_file, segment_start, segment_end, FRAME_SAMPLING_RATE, max_retries=3)

            if video_embedding:
                video_file_name = os.path.basename(video_file)
                video_file_path = os.path.dirname(video_file)
                
                # Prepare data for insertion (Milvus requires columns)
                vector_data = [{
                        "video_id": idx,
                        "video_file_name": video_file_name,
                        "video_file_path": video_file_path,
                        "segment_start": segment_start,
                        "segment_end": segment_end, 
                        "clip_vector": video_embedding,
                    }]
                
                # Insert the data into sqlite
                insert_result_sqlite = save_to_db(video_file_name, video_file_path, segment_start, segment_end, annotate_res)
                print(insert_result_sqlite)
                print(f"\nInsertion successful. result: {insert_result_sqlite}")
                
                # Insert the data into Milvus
                insert_result_milvus = client.insert(collection_name=COLLECTION_NAME, data=vector_data)
                print(insert_result_milvus)
                print(f"\nInsertion successful. result: {insert_result_milvus}")


if __name__ == "__main__":
    train()
