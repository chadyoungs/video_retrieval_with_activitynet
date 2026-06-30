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

# semantic model
SEMANTIC_MODEL_TIER = "base"  # options: base, advanced
BASE_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
ADVANCED_CLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"
EMBEDDING_DIM_BASE = 512
EMBEDDING_DIM_ADVANCED = 1152

if SEMANTIC_MODEL_TIER == "advanced":
    CLIP_MODEL_NAME = ADVANCED_CLIP_MODEL_NAME
    EMBEDDING_DIM = EMBEDDING_DIM_ADVANCED
else:
    CLIP_MODEL_NAME = BASE_CLIP_MODEL_NAME
    EMBEDDING_DIM = EMBEDDING_DIM_BASE

EMBEDDING_MODEL_VERSION = CLIP_MODEL_NAME
ANNOTATION_MODEL_VERSION = OLLAMA_MODEL

# dataset source options: activitynet, nuscenes
DATASET_SOURCE = "activitynet"

# activitynet inputs
ACTIVITYNET_DATA_ROOT = "/mnt/sdc/activitynet_caption/demo_data"
ACTIVITYNET_FILE_FORMAT = "mp4"

# nuScenes generated-video inputs
NUSCENES_VIDEO_ROOT = "/mnt/sdc/nuscenes/videos/cam_front"
NUSCENES_CLIP_MAPPING_FILE = "/mnt/sdc/nuscenes/videos/cam_front/clip_mapping.json"
NUSCENES_DATASET_NAME = "nuscenes"
NUSCENES_DATASET_VERSION = "v1.0-trainval"
NUSCENES_DATASET_SPLIT = "train"
NUSCENES_CAMERA_CHANNEL = "CAM_FRONT"

# data
CLIP_DURATION = 10  # seconds
FRAME_SAMPLING_RATE = 10  # frame sampling

# computation
BATCH_VIDEO = 2  # number of videos per outer batch
BATCH_SIZE_DB = 5  # batch size for database insertion
# NUM_PROCESSES = max(1, multiprocessing.cpu_count() // 2)  # parallel video workers
NUM_WORKERS = 1  # for torch
NUM_PROCESSES = 2  # parallel video workers, local model can't handle >2 processes

# Maximum number of frames sent to the VLM for scene annotation per segment.
# Sending 4 keyframes instead of all sampled frames dramatically reduces VLM
# input size while preserving annotation quality.
N_VLM_FRAMES = 4

# Number of candidate segments to retrieve before VLM reranking.
RERANKER_CANDIDATES = 10
