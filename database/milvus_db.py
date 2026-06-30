import sys
from pathlib import Path
import re

sys.path.append(str(Path(__file__).parent.parent))

from pymilvus import DataType, MilvusClient

from utils.config import ALIAS, COLLECTION_NAME, EMBEDDING_DIM, MILVUS_HOST, MILVUS_PORT

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


def create_milvus_collection(client, collection_name, dim):
    if client.has_collection(collection_name, using=ALIAS):
        client.drop_collection(collection_name, using=ALIAS)

    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
        description="Video CLIP Embeddings Index",
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="video_file_name", datatype=DataType.VARCHAR, max_length=256
    )
    schema.add_field(
        field_name="video_file_path", datatype=DataType.VARCHAR, max_length=512
    )
    schema.add_field(field_name="segment_start", datatype=DataType.FLOAT)
    schema.add_field(field_name="segment_end", datatype=DataType.FLOAT)
    schema.add_field(field_name="dataset_name", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(
        field_name="camera_channel", datatype=DataType.VARCHAR, max_length=64
    )
    schema.add_field(field_name="clip_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="scene_token", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(
        field_name="embedding_model", datatype=DataType.VARCHAR, max_length=256
    )
    schema.add_field(field_name="clip_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="clip_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )

    print(f"Collection '{collection_name}' initialized successfully.")


def batch_insert_milvus(client, collection_name, batch_data):
    if len(batch_data) == 0:
        return None

    return client.insert(collection_name=collection_name, data=batch_data)


def _build_filter_expression(metadata_filters: dict):
    if not metadata_filters:
        return None

    allowed_fields = {
        "dataset_name",
        "camera_channel",
        "clip_id",
        "scene_token",
        "embedding_model",
    }

    clauses = []
    for key, value in metadata_filters.items():
        if key not in allowed_fields:
            continue
        if not re.match(r"^[\w\-.:/ ]+$", str(value)):
            continue
        safe_value = str(value).replace("\\", "\\\\").replace("'", "\\'")
        clauses.append(f"{key} == '{safe_value}'")

    if not clauses:
        return None
    return " and ".join(clauses)


def search_milvus(
    client, query_embedding: list, limit: int = 5, metadata_filters: dict = None
) -> list:
    try:
        filter_expr = _build_filter_expression(metadata_filters)
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=limit,
            output_fields=[
                "video_file_name",
                "video_file_path",
                "segment_start",
                "segment_end",
                "dataset_name",
                "camera_channel",
                "clip_id",
                "scene_token",
                "embedding_model",
            ],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            filter=filter_expr,
        )

        milvus_hits = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            milvus_hits.append(
                {
                    "video_file_name": entity.get("video_file_name"),
                    "video_file_path": entity.get("video_file_path"),
                    "segment_start": entity.get("segment_start"),
                    "segment_end": entity.get("segment_end"),
                    "dataset_name": entity.get("dataset_name"),
                    "camera_channel": entity.get("camera_channel"),
                    "clip_id": entity.get("clip_id"),
                    "scene_token": entity.get("scene_token"),
                    "embedding_model": entity.get("embedding_model"),
                    "distance": hit.get("distance"),
                    "score": 1 - hit.get("distance", 0),
                }
            )
        return milvus_hits

    except Exception as e:
        print(f"Milvus search error: {e}")
        return []


if __name__ == "__main__":
    client = get_milvus_client()
    create_milvus_collection(client, COLLECTION_NAME, EMBEDDING_DIM)
