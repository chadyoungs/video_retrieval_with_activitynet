import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

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
    schema.add_field(field_name="video_file_name", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="video_file_path", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="segment_start", datatype=DataType.FLOAT)
    schema.add_field(field_name="segment_end", datatype=DataType.FLOAT)
    schema.add_field(field_name="clip_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="clip_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 8, "efConstruction": 200}
    )

    # 5. Create Collection (This creates AND loads it automatically!)
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params
    )

    print(f"Collection '{collection_name}' initialized successfully.")


def batch_insert_milvus(client, collection_name, batch_data):
    if len(batch_data) == 0:
        return None
    return client.insert(collection_name=collection_name, data=batch_data)


if __name__ == "__main__":
    client = get_milvus_client()
    create_milvus_collection(client, COLLECTION_NAME, EMBEDDING_DIM)
