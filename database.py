from pymilvus import MilvusClient, DataType, FieldSchema, Collection, CollectionSchema, connections, utility

from config import MILVUS_URI, MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME
from embedding import EMBEDDING_DIM

# Initialize Milvus client
milvus_client = MilvusClient(uri=MILVUS_URI)

ALIAS = "default" # Use 'default' unless you have a specific reason otherwise

# Establish the connection
try:
    connections.connect(
        alias=ALIAS,
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    print(f"Successfully connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT} using alias '{ALIAS}'.")

except Exception as e:
    # Handle the error, perhaps exit the script
    print(f"Failed to connect to Milvus: {e}")
    

# MILVUS COLLECTION DEFINITION (To be run once) ---
def create_milvus_collection(client, collection_name, dim):
    """
    Milvus collection definition
    """
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
        FieldSchema(name="video_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="video_filepath", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="clip_vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, enable_dynamic_field=True, description="Video Reverse Search Index with CLIP embeddings")
    
    collection = Collection(name=collection_name, schema=schema)

    index_params = MilvusClient.prepare_index_params(
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 8, "efConstruction": 200}
    )
    
    collection.create_index(field_name='clip_vector', index_params=index_params)

    print(f"Collection '{collection_name}' created successfully with dimension {dim}.")


if __name__ == "__main__":
    create_milvus_collection(milvus_client, COLLECTION_NAME, EMBEDDING_DIM)