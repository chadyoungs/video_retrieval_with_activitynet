import os

from PIL import Image
import cv2
import torch

from pymilvus import connections, Collection

from database import milvus_client
from embedding import processor, model
from config import COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT

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


def retrieval_with_text(query_text):
    text_inputs = processor(text=query_text, return_tensors="pt")
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    normalized_query = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    print(f"\nSearching Milvus for: '{query_text}'...")

    collection = Collection(COLLECTION_NAME)
    collection.load()

    search_results = collection.search(
        data=[normalized_query.squeeze().tolist()],
        limit=5,
        output_fields=["video_filepath"],
        anns_field="clip_vector",
        param={"metric_type": "COSINE"},
    )

    print("--- Search Results (Top 1) ---")

    if search_results:
        # Note: The search returns distance/shmdb5imilarity, you can filter this.
        top_hit = search_results[0][-1]
        print(f"Video: {top_hit['entity']['video_filepath']}")
        print(f"Distance (Lower is better): {top_hit['distance']}")


def retrieval_with_image(query_img):
    img_inputs = processor(images=query_img, return_tensors="pt")
    with torch.no_grad():
        img_features = model.get_image_features(**img_inputs)

    normalized_query = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

    print(f"\nSearching Milvus for: '{query_text}'...")

    collection = Collection(COLLECTION_NAME)
    collection.load()

    search_results = collection.search(
        data=[normalized_query.squeeze().tolist()],
        limit=5,
        output_fields=["video_filepath"],
        anns_field="clip_vector",
        param={"metric_type": "COSINE"},
    )

    print("--- Search Results (Top 1) ---")
    if search_results:
        # Note: The search returns distance/shmdb5imilarity, you can filter this.
        top_hit = search_results[0][-1]
        print(f"Video: {top_hit['entity']['video_filepath']}")
        print(f"Distance (Lower is better): {top_hit['distance']}")


if __name__ == "__main__":
    query_text = "hand clapping"
    retrieval_with_text(query_text)

    query_img = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./samples/person01_handclapping_d1_uncomp_sample.png",
    )
    retrieval_with_image(query_img)
