import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from pymilvus import Collection, MilvusClient, connections

from database.sql_db import query_annotation_by_conditions
from utils.config import (
    ALIAS,
    CLIP_DURATION,
    COLLECTION_NAME,
    FRAME_SAMPLING_RATE,
    MILVUS_HOST,
    MILVUS_PORT,
)
from utils.embedding import EMBEDDING_DIM, model, processor

milvus_client = None
try:
    milvus_client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}", alias=ALIAS)
    connections.connect(alias=ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
    print(
        f"Successfully connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT} (alias: {ALIAS})"
    )
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")
    sys.exit(1)


def normalize_feature(features: torch.Tensor) -> List[float]:
    normalized = features / features.norm(p=2, dim=-1, keepdim=True)
    return normalized.squeeze().cpu().numpy().tolist()


def get_text_embedding(query_text: str) -> List[float]:
    text_inputs = processor(
        text=query_text, return_tensors="pt", padding=True, truncation=True
    )
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs).pooler_output
    return normalize_feature(text_features)


def get_image_embedding(query_img_path: str) -> List[float]:
    if isinstance(query_img_path, str):
        img = Image.open(query_img_path).convert("RGB")
    elif isinstance(query_img_path, Image.Image):
        img = query_img_path
    else:
        raise ValueError("query_img must be path str or PIL.Image")

    img_inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        img_features = model.get_image_features(**img_inputs).pooler_output
    return normalize_feature(img_features)


def search_milvus(
    query_embedding: List[float], limit: int = 5, metric_type: str = "COSINE"
) -> List[Dict]:

    try:
        collection = Collection(COLLECTION_NAME, alias=ALIAS)
        collection.load()

        search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="clip_vector",
            param=search_params,
            limit=limit,
            output_fields=[
                "video_id",
                "video_file_name",
                "video_file_path",
                "segment_start",
                "segment_end",
            ],
        )
        collection.release()

        milvus_hits = []
        for hit in results[0]:
            entity = hit.entity
            milvus_hits.append(
                {
                    "video_file_name": entity.get("video_file_name"),
                    "video_file_path": entity.get("video_file_path"),
                    "segment_start": entity.get("segment_start"),
                    "segment_end": entity.get("segment_end"),
                    "distance": hit.distance,
                    "score": (
                        1 - hit.distance if metric_type == "COSINE" else hit.distance
                    ),
                }
            )
        return milvus_hits
    except Exception as e:
        print(f"Milvus search error: {e}")
        return []


def search_sql(annotation_conditions: Dict, limit: int = 5) -> List[Dict]:
    try:
        sql_hits = query_annotation_by_conditions(
            conditions=annotation_conditions, limit=limit
        )
        for hit in sql_hits:
            hit["score"] = 1.0
        return sql_hits
    except Exception as e:
        print(f"SQL search error: {e}")
        return []


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
    # text based
    query_text = "hand clapping"
    annotation_conditions = {"scene_env": "outdoor", "person_count": "few"}
    retrieval_with_text(query_text, annotation_conditions, limit=5)

    # image based
    query_img = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "./samples/person01_handclapping_d1_uncomp_sample.png",
    )
    retrieval_with_image(
        query_img_path=query_img,
        annotation_conditions={"weather": "sunny", "lighting": "bright"},
        limit=5,
    )

    # video based
    # To do
