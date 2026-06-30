# Video Retrieval with ActivityNet / nuScenes
> Hybrid retrieval over video segments with semantic embeddings + structured metadata.

## Local resources
- NVIDIA-GeForce RTX 3060 Ti, 8G

## Usage
- Recommend using Anaconda to activate a virtual environment.
- Python version: **3.10.19**
- Install dependencies:
  - `pip install -r requirements.txt`

## End-to-end pipeline
1. Initialize storage:
   - `python ./database/milvus_db.py`
   - `python ./database/sql_db.py`
2. (Optional for nuScenes) Generate CAM_FRONT videos from image frames:
   - `python ./utils/nuscenes_video_builder.py --dataroot <NUSCENES_DATA_ROOT> --meta-root <NUSCENES_META_JSON_ROOT> --output-dir <OUTPUT_VIDEO_DIR> --dataset-version v1.0-trainval --dataset-split train --fps 12`
   - This creates MP4 clips and `clip_mapping.json` (clip-to-source traceability metadata).
3. Configure source in `utils/config.py`:
   - `DATASET_SOURCE = "activitynet"` or `"nuscenes"`
   - For nuScenes set `NUSCENES_VIDEO_ROOT` and `NUSCENES_CLIP_MAPPING_FILE`.
4. Build embeddings + annotations + metadata index:
   - `python train.py`
5. Run retrieval:
   - `python retrieval.py`

## Solution
The architecture has five components:
1. **Dataset adapter layer** (`utils/dataset_adapter.py`): provides a common video-entry interface for ActivityNet and generated nuScenes clips.
2. **Feature extraction / annotation** (`utils/embedding.py`): VLM tags include environment tags + driving-specific semantics (`driving_context`, `road_user_density`, `traffic_flow`, `semantic_summary`).
3. **Embedding extraction** (`utils/embedding.py`): supports model tier selection via config (`base` vs `advanced`) with model-versioned metadata.
4. **Storage**:
   - Milvus: vector retrieval + key scalar context (`dataset_name`, `camera_channel`, `clip_id`, `scene_token`, `embedding_model`).
   - SQLite: rich metadata for filtering/ranking.
5. **Hybrid retrieval** (`retrieval.py`): vector similarity fused with SQL metadata scoring, plus dataset-aware filtering.

## Metadata schema
- Canonical schema: `database/video_metadata_schema.json`
- Covers:
  - identity (dataset/version/split/clip/segment)
  - temporal fields (clip + segment)
  - sensor/context fields (`CAM_FRONT`, scene/log/location/map)
  - semantics (existing + driving-specific tags + summary)
  - retrieval lineage (embedding/annotation model versions, processing timestamp, quality flags)

## Minimal validation checklist (before full-scale indexing)
- [ ] `clip_mapping.json` exists and each entry points to a real clip file.
- [ ] `camera_channel` is always `CAM_FRONT` for nuScenes clips.
- [ ] Metadata required fields match `database/video_metadata_schema.json`.
- [ ] Sample retrieval with `metadata_filters={"dataset_name": "nuscenes", "camera_channel": "CAM_FRONT"}` returns valid clips.
- [ ] Embedding model/version and annotation model/version are stored in SQLite rows.

## Appendix
### How to download huggingface model in China
- Step1: `pip install -U huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/`
- Step2: `export HF_ENDPOINT="https://hf-mirror.com"`
- Step3: e.g. `hf download google-bert/bert-base-chinese`

### local Milvus related
- install milvus by following the official guide
- dataset visualization: **Attu**

### cuda
- models and data should load on cuda when GPU is available
