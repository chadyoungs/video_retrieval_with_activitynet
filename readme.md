# Video Retrieval with ActivityNet

## Usage
- Step1. ``python ./database/milvus_db.py`` & ``python ./database/sql_db.py`` to setup the milvus database & sqlite database
- Step2. ``python train.py`` to collect embeddings & annotate of training data
- Step3. ``python retrieval.py``
  - retrieval option, text or image or video

## Principle
The hybrid retrieval approach combines both content-based and semantic-based retrieval methods to achieve improved accuracy and efficiency. The system utilizes visual features extracted from video frames along with semantic embeddings of actions and objects within the videos. This enables the application to retrieve videos that match user queries more effectively.

## Principle
The hybrid retrieval approach combines both content-based and semantic-based retrieval methods to achieve improved accuracy and efficiency. The system utilizes visual features extracted from video frames along with semantic embeddings of actions and objects within the videos. This enables the application to retrieve videos that match user queries more effectively.

## Solution
The system architecture consists of three main components:
1. **Feature Extraction:** Visual features are extracted from videos using ViT. Besides, attributes of video was detected by VLM. 

2. **Retrieval Engine:** The retrieval engine matches user queries against the stored video features using a hybrid approach, returning the most relevant results based on both visual and semantic similarity.

This comprehensive solution allows users to perform efficient and accurate video retrieval based on a wide range of queries.

## Appendix

### How to download huggingface model in China
- Step1. pip install -U huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/ 

- Step2. export HF_ENDPOINT="https://hf-mirror.com" 

- Step3. e.g. hf download google-bert/bert-base-chinese

### local Milvus related
- install milvus by following the official Guide
- Dataset visualization: **Attu**
