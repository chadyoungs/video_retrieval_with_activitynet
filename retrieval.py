import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from typing import Dict, List, Tuple

import torch
from PIL import Image

from database.milvus_db import get_milvus_client, search_milvus
from database.sql_db import search_sql
from utils.config import (
    CLIP_DURATION,
    COLLECTION_NAME,
    FRAME_SAMPLING_RATE,
    MILVUS_HOST,
    MILVUS_PORT,
)
from utils.embedding import DEVICE, EMBEDDING_DIM, get_model, get_processor

# Obtain (or create) the shared Milvus singleton.  This replaces the previous
# pattern of creating a second MilvusClient + connections.connect() pair here,
# which resulted in two separate connections to the same server.
try:
    get_milvus_client()
    print(f"Successfully connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
    sys.exit(1)


def normalize_feature(features: torch.Tensor) -> List[float]:
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)
    return normalized.squeeze().cpu().numpy().tolist()


def get_text_embedding(query_text: str) -> List[float]:
    _processor = get_processor()
    _model = get_model()
    text_inputs = _processor(
        text=query_text, return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        text_features = _model.get_text_features(**text_inputs).pooler_output
    return normalize_feature(text_features)


def get_image_embedding(query_img_path: str) -> List[float]:
    if isinstance(query_img_path, str):
        img = Image.open(query_img_path).convert("RGB")
    elif isinstance(query_img_path, Image.Image):
        img = query_img_path
    else:
        raise ValueError("query_img must be path str or PIL.Image")

    _processor = get_processor()
    _model = get_model()
    img_inputs = _processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        img_features = _model.get_image_features(**img_inputs).pooler_output
    return normalize_feature(img_features)


def hybrid_retrieval(
    query_embedding: List[float],
    annotation_conditions: Dict = None,
    milvus_limit: int = 10,
    sql_limit: int = 10,
    weight_milvus: float = 0.7,
    weight_sql: float = 0.3,
) -> List[Dict]:
    """
    hybrid search: Milvus vector + SQL tag fusion
    :param query_embedding: query embedding (text/image)
    :param annotation_conditions: SQL annotation conditions (e.g., {"scene_type": "street", "person_count": "crowd"})
    :param milvus_limit: Milvus search limit
    :param sql_limit: SQL search limit
    :param weight_milvus: weight for Milvus score
    :param weight_sql: weight for SQL score
    :return: fused retrieval results (sorted by final score in descending order)
    """
    # 1. Execute Milvus vector retrieval
    milvus_hits = search_milvus(query_embedding, limit=milvus_limit)
    if not milvus_hits:
        print("Milvus search returned empty results")

    # 2. Execute SQL tag retrieval (skip if no conditions)
    sql_hits = []
    if annotation_conditions and len(annotation_conditions) > 0:
        sql_hits = search_sql(annotation_conditions, limit=sql_limit)
        if not sql_hits:
            print(
                "SQL search returned empty results for conditions:",
                annotation_conditions,
            )

    # 3. Fuse results (deduplicate based on video_file_name + segment_start and apply weighted scoring)
    result_map = {}

    # Process Milvus results
    for hit in milvus_hits:
        key = f"{hit['video_file_name']}_{hit['segment_start']}"
        result_map[key] = {
            **hit,
            "milvus_score": hit["score"],
            "sql_score": 0.0,  # Initial SQL score is 0
            "final_score": hit["score"] * weight_milvus,
        }

    # Process SQL results (update scores if matched)
    for hit in sql_hits:
        key = f"{hit['video_file_name']}_{hit['segment_start']}"
        if key in result_map:
            # Existing Milvus result, add SQL score
            result_map[key]["sql_score"] = hit["score"]
            result_map[key]["final_score"] += hit["score"] * weight_sql
        else:
            # Only SQL match, Milvus score is 0
            result_map[key] = {
                **hit,
                "milvus_score": 0.0,
                "sql_score": hit["score"],
                "final_score": hit["score"] * weight_sql,
            }

    # 4. Sort by final score in descending order, return top N results
    final_results = sorted(
        result_map.values(), key=lambda x: x["final_score"], reverse=True
    )
    return final_results[: max(milvus_limit, sql_limit)]


def retrieval_with_text(
    query_text: str, annotation_conditions: Dict = None, limit: int = 5
) -> List[Dict]:
    """
    Hybrid text retrieval: text vector + SQL annotation tags
    :param query_text: query text
    :param annotation_conditions: SQL annotation conditions (e.g., {"scene_env": "outdoor", "weather": "sunny"})
    :param limit: number of results to return
    :return: fused retrieval results
    """
    print(f"\n=== Hybrid Text Retrieval for: '{query_text}' ===")
    print(f"SQL annotation conditions: {annotation_conditions or 'None'}")

    text_embedding = get_text_embedding(query_text)

    hybrid_results = hybrid_retrieval(
        query_embedding=text_embedding,
        annotation_conditions=annotation_conditions,
        milvus_limit=limit * 2,
        sql_limit=limit * 2,
    )[:limit]

    print("--- Top Retrieval Results ---")
    for idx, res in enumerate(hybrid_results, 1):
        print(f"\nRank {idx}:")
        print(f"  Video: {res['video_file_path']}/{res['video_file_name']}")
        print(f"  Segment: {res['segment_start']}s - {res['segment_end']}s")
        print(f"  Milvus Score: {res['milvus_score']:.4f}")
        print(f"  SQL Score: {res['sql_score']:.4f}")
        print(f"  Final Score: {res['final_score']:.4f}")
        print(f"  Distance (Milvus): {res.get('distance', 'N/A'):.4f}")

    return hybrid_results


def retrieval_with_image(
    query_img_path: str, annotation_conditions: Dict = None, limit: int = 5
) -> List[Dict]:
    """
    Hybrid image retrieval: image vector + SQL annotation tags
    :param query_img_path: query image path
    :param annotation_conditions: SQL annotation conditions
    :param limit: number of results to return
    :return: fused retrieval results
    """
    if not os.path.exists(query_img_path):
        print(f"Error: Image file not found - {query_img_path}")
        return []

    print(f"\n=== Hybrid Image Retrieval for: '{query_img_path}' ===")
    print(f"SQL annotation conditions: {annotation_conditions or 'None'}")

    img_embedding = get_image_embedding(query_img_path)

    hybrid_results = hybrid_retrieval(
        query_embedding=img_embedding,
        annotation_conditions=annotation_conditions,
        milvus_limit=limit * 2,
        sql_limit=limit * 2,
    )[:limit]

    print("--- Top Retrieval Results ---")
    for idx, res in enumerate(hybrid_results, 1):
        print(f"\nRank {idx}:")
        print(f"  Video: {res['video_file_path']}/{res['video_file_name']}")
        print(f"  Segment: {res['segment_start']}s - {res['segment_end']}s")
        print(f"  Milvus Score: {res['milvus_score']:.4f}")
        print(f"  SQL Score: {res['sql_score']:.4f}")
        print(f"  Final Score: {res['final_score']:.4f}")
        print(f"  Distance (Milvus): {res.get('distance', 'N/A'):.4f}")

    return hybrid_results


if __name__ == "__main__":
    option = "text"

    if option == "text":
        query_text = "snowy mountain"
        annotation_conditions = {"scene_env": "outdoor", "person_count": "single"}
        retrieval_with_text(query_text, annotation_conditions, limit=5)
    elif option == "image":
        query_img = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "./samples/v_0_1BQPWzRiw_sample.png",
        )
        retrieval_with_image(
            query_img_path=query_img,
            annotation_conditions={"scene_env": "outdoor", "lighting": "bright"},
            limit=5,
        )
    else:
        # video based
        # To do
        pass
