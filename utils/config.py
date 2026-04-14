import multiprocessing

# database
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

COLLECTION_NAME = "video_reverse_search_demo_index"
ALIAS = "activitynet-demo"

# ollama
OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3-vl:8b"
OLLAMA_TIMEOUT = 600

MAX_RETRIES = 3  # actually 1, for saving time
RETRY_BACKOFF_FACTOR = 1.5

# model
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# data
CLIP_DURATION = 10  # seconds
FRAME_SAMPLING_RATE = 25  # frame sampling

# computation
NUM_WORKERS = 4
BATCH_SIZE_DB = 200  # batch size for database insertion
NUM_PROCESSES = int(multiprocessing.cpu_count() / 2)

CLIP_BATCH_SIZE = 6  # batch size for CLIP embedding generation
