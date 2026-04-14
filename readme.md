## Usage
- Step1. python ./database/milvus_db.py & sql_db.py to setup the milvus database & sqlite database
- Step2. python train.py to collect embeddings & annotate of training data
- Step3. **python retrieval.py**
  - retrieval option, text or image or video


## How to download huggingface model in China
- Step1. pip install -U huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/ 

- Step2. export HF_ENDPOINT="https://hf-mirror.com" 

- Step3. e.g. hf download google-bert/bert-base-chinese

## local Milvus related
- install milvus by following the official Guide
- UI: localhost:9091/webui
