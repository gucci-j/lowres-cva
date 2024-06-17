Vocabulary Expansion for Low-resource Cross-lingual Transfer
===

This is the official code for the paper titled "Vocabulary Expansion for Low-resource Cross-lingual Transfer." For reproduction, please refer to [Reproduction](#reproduction).

## Requirements
* Python 3.11.7 or later
* CUDA 11.8
* torch==2.2.1
* transformers==4.39.0.dev0
* jupyterlab==4.1.2
* peft==0.8.2
* datasets==2.17.1
* evaluate==0.4.1
* bitsandbytes==0.42.0
* scikit-learn==1.4.1.post1
* seaborn==0.13.2
* sumeval==0.2.2
* janome==0.5.0
* protobuf==4.25.1
* entmax==1.3
* fastdist==1.1.6
* rouge-score==0.1.2
* numba==0.59.0
* tensorboardX==2.6.2.2
* tensorboard==2.16.2
* torch_tb_profiler==0.4.3
* pyarabic==0.6.15
* rouge==1.0.1
* huggingface-hub==0.21.1
* zstandard==0.22.0
* lm_eval==0.4.2
* lighteval==0.4.0
* openai==1.25.0
* tiktoken==0.6.0
* fasttext==0.9.2 (See below)

## Installation
After manually installing `PyTorch` and `transformers`, please run the following.
```bash
pip install -r requirements.txt
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
```

## Reproduction
### 1. Preprocessing
See [Preprocessing](./preprocessing/README.md).

### 2. Target Model Initialization
See [Instantiation](./instantiation/README.md)

### 3. Language Adaptive Pre-training
See [LAPT](./lapt/README.md)

### 4. Evaluation
See [Evaluation](./eval/README.md)

## Adapted Models
All models will be available on the Hugging Face Model Hub.

## License
[MIT License](./LICENSE)
