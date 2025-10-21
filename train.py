import os
import torch  # PyTorch is needed for CLIP model inference
from pymilvus import Collection, connections

from database import milvus_client
from embedding import generate_video_embedding, EMBEDDING_DIM
from config import FRAME_SAMPLING_RATE, COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT


ALIAS = "default"  # Use 'default' unless you have a specific reason otherwise

# Establish the connection
try:
    connections.connect(alias=ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
    print(
        f"Successfully connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT} using alias '{ALIAS}'."
    )

except Exception as e:
    # Handle the error, perhaps exit the script
    print(f"Failed to connect to Milvus: {e}")

collection = Collection(COLLECTION_NAME)


def get_video_file_demo_list():
    root = "/ext-data/datasets/training_lib_KTH"
    file_list = [
        os.path.join(root, file) for file in os.listdir(root) if file.endswith("avi")
    ]

    return file_list


def train():
    video_file_list = get_video_file_demo_list()

    # Process the video dataset
    for idx, video_file in enumerate(video_file_list):
        video_embedding = generate_video_embedding(
            video_file, FRAME_SAMPLING_RATE, EMBEDDING_DIM
        )

        if video_embedding:
            # Prepare data for insertion (Milvus requires columns)
            data = [
                {
                    "video_id": idx,
                    "video_filepath": video_file,
                    "clip_vector": video_embedding,
                }
            ]

            # Insert the data into Milvus
            insert_result = collection.insert(data)
            print(insert_result)
            print(f"\nInsertion successful. result: {insert_result}")


if __name__ == "__main__":
    train()
