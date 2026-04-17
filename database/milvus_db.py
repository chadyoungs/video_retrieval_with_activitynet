import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pymilvus import DataType, MilvusClient

from utils.config import ALIAS, COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT
from utils.embedding import EMBEDDING_DIM

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

    # Add fields
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="video_file_name", datatype=DataType.VARCHAR, max_length=256
    )
    schema.add_field(
        field_name="video_file_path", datatype=DataType.VARCHAR, max_length=256
    )
    schema.add_field(field_name="segment_start", datatype=DataType.FLOAT)
    schema.add_field(field_name="segment_end", datatype=DataType.FLOAT)
    schema.add_field(field_name="clip_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="clip_vector",
        index_type="HNSW",
        metric_type="COSINE",
        # M=16 is the recommended default (M=8 degrades recall for large collections).
        # efConstruction=200 keeps index build quality high.
        params={"M": 16, "efConstruction": 200},
    )

    # 5. Create Collection (This creates AND loads it automatically!)
    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )

    print(f"Collection '{collection_name}' initialized successfully.")


def batch_insert_milvus(client, collection_name, batch_data):
    if len(batch_data) == 0:
        return None

    return client.insert(collection_name=collection_name, data=batch_data)


def search_milvus(client, query_embedding: list, limit: int = 5) -> list:
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],  # List of vectors
            limit=limit,
            output_fields=[
                "video_file_name",
                "video_file_path",
                "segment_start",
                "segment_end",
            ],
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        )

        milvus_hits = []
        # results is a list of results (one per query vector)
        for hit in results[0]:
            # 'entity' in the new API is just a dictionary
            entity = hit.get("entity", {})
            milvus_hits.append(
                {
                    "video_file_name": entity.get("video_file_name"),
                    "video_file_path": entity.get("video_file_path"),
                    "segment_start": entity.get("segment_start"),
                    "segment_end": entity.get("segment_end"),
                    "distance": hit.get("distance"),
                    # For COSINE, higher score usually means more similar
                    "score": 1
                    - hit.get("distance", 0),  # Convert distance to similarity score
                }
            )
        return milvus_hits

    except Exception as e:
        print(f"Milvus search error: {e}")
        return []


if __name__ == "__main__":
    client = get_milvus_client()
    create_milvus_collection(client, COLLECTION_NAME, EMBEDDING_DIM)
