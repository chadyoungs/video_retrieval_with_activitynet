import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from utils.config import ALIAS, COLLECTION_NAME, MILVUS_HOST, MILVUS_PORT
from utils.embedding import EMBEDDING_DIM

# Establish the connection
try:
    connections.connect(alias=ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
    print(
        f"Successfully connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT} using alias '{ALIAS}'."
    )

except Exception as e:
    # Handle the error, perhaps exit the script
    print(f"Failed to connect to Milvus: {e}")


# MILVUS COLLECTION DEFINITION (To be run once) ---
def create_milvus_collection(collection_name, dim):
    """
    Milvus collection definition
    """
    if utility.has_collection(collection_name, using=ALIAS):
        utility.drop_collection(collection_name, using=ALIAS)

    fields = [
        FieldSchema(
            name="video_id", dtype=DataType.INT64, is_primary=True, auto_id=False
        ),
        FieldSchema(name="video_file_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="video_file_path", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="segment_start", dtype=DataType.FLOAT),
        FieldSchema(name="segment_end", dtype=DataType.FLOAT),
        FieldSchema(name="clip_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(
        fields,
        enable_dynamic_field=True,
        description="Video CLIP Embeddings Index",
    )

    collection = Collection(name=collection_name, schema=schema, using=ALIAS)

    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 8, "efConstruction": 200},
    }

    collection.create_index(field_name="clip_vector", index_params=index_params)

    print(f"Collection '{collection_name}' created successfully with dimension {dim}.")


def batch_insert_milvus(client, collection_name, batch_data):
    if len(batch_data) == 0:
        return None
    return client.insert(collection_name=collection_name, data=batch_data)


if __name__ == "__main__":
    create_milvus_collection(COLLECTION_NAME, EMBEDDING_DIM)
