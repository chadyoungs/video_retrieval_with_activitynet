# Video Retrieval with ActivityNet
> A VLM application specified on video retrieval with ActivityNet Datase
## Local resources
- NVIDIA-GeForce RTX 3060 Ti, 8G

## Usage
- Recommend using Anaconda to activate a virtual environment
  - Python Version, **3.10.19**

- ``pip install -r requiremnts.txt`` at first

- Step1. ``python ./database/milvus_db.py`` & ``python ./database/sql_db.py`` to setup the milvus database & sqlite database
- Step2. ``python train.py`` to collect embeddings & annotate of training data
- Step3. ``python retrieval.py``
  - retrieval option, text or image or video

## Principle
The hybrid retrieval approach combines both keywords-based and semantic-based retrieval methods to achieve improved accuracy and efficiency. The system utilizes visual features (keywords, e.g. scene type, weather and so on) extracted from video frames along with semantic embeddings of the videos. This enables the application to retrieve videos that match user queries more effectively.

## Solution
The system architecture consists of three main components:
1. **Feature Extraction:** Visual features (keywords) are extracted from videos using VLM (qwen3-vl:8b model is used here based on local ollama service).

2. **Embedding:** Semantic embedding are extracted from videos using Clip (clip-vit-base-patch32 is used here).

3. **Video segmentation:** Videos are segmented into sub-videos with duration of 10 seconds.
    
4. **keyframes selection:** To save compute resources, key frames (4 frames are used here, integrating three key frame selection strategies) are selected to represent a video segment.

5. **Retrieval Engine:** The retrieval engine matches user queries against the stored video features using a hybrid approach, returning the most relevant results based on both visual and semantic similarity.

This comprehensive solution allows users to perform efficient and accurate video retrieval based on a wide range of queries.

## Appendix

### How to download huggingface model in China
- Step1. pip install -U huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/ 

- Step2. export HF_ENDPOINT="https://hf-mirror.com" 

- Step3. e.g. hf download google-bert/bert-base-chinese

### local Milvus related
- install milvus by following the official Guide
- Dataset visualization: **Attu**

### local ollama

### cuda
- model and data should loaded with cuda while GPU is available (with more GPU resources)
