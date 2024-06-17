Preprocessing
===

## Reproduction
### 1. Download CC-100 data
Download the CC-100 data for seven languages: Arabic, German, Greek, Hindi, Japanese, Swahili, and Thai. They are available at https://data.statmt.org/cc-100/

### 2. Extract data
Randomly extract 2^20 and 2^15 sentences for high- and low-resource settings for each language.

**Example**: Extracting data for German in high-resource settings.  
```bash
i=20
lang_code=de
output_dir=/path/to/${lang_code}_2^${i}/
mkdir $output_dir
input_file="/path/to/cc100/${lang_code}.txt"
output_file="/path/to/${lang_code}_2^${i}_extracted.txt"
num_sentences=$((2**$i))
shuf -n "$num_sentences" "$input_file" > "$output_file"
```

### 3. Train tokenizers
Train a tokenizer for each setting on the extracted texts obtained in 2.

#### Usage
```
$ python src/train_tokenizer.py -h
usage: train_tokenizer.py [-h] --corpus_path CORPUS_PATH --vocab_size
                          VOCAB_SIZE --source_tokenizer_path
                          SOURCE_TOKENIZER_PATH --output_dir OUTPUT_DIR
                          --lang_code {ar,ja,de,sw,th,hi,el,gn}
                          [--num_new_tokens NUM_NEW_TOKENS]
                          [--num_max_iter NUM_MAX_ITER]

options:
  -h, --help            show this help message and exit
  --corpus_path CORPUS_PATH
                        Path to the corpus to train the tokenizer on
  --vocab_size VOCAB_SIZE
                        Vocabulary size of the tokenizer
  --source_tokenizer_path SOURCE_TOKENIZER_PATH
                        Path to the source tokenizer
  --output_dir OUTPUT_DIR
                        Path to the output directory
  --lang_code {ar,ja,de,sw,th,hi,el,gn}
                        Language code
  --num_new_tokens NUM_NEW_TOKENS
                        Number of new tokens to add to the tokenizer
  --num_max_iter NUM_MAX_ITER
                        Maximum number of iterations to remove unused tokens
```

#### Example
The following trains a German tokenizer built on top of the LLaMA2 tokenizer with $\mathcal{V}_\text{new}=100$ and $\mathcal{V}_\text{aux}=50000$
```bash
lang_code=de
vocab_size=50000
input_file="/path/to/${lang_code}_2^${i}_extracted.txt"
output_dir="/path/to/${lang_code}_2^${i}/"

python train_tokenizer.py \
    --corpus_path ${input_file} \
    --vocab_size ${vocab_size} \
    --source_tokenizer_path /path/to/models--meta-llama--Llama-2-7b-hf/snapshots/put_snapshot_id_here/tokenizer.model \
    --output_dir ${output_dir} \
    --lang_code ${lang_code} \
    --num_new_tokens 100
```

### 4. Generate LAPT data
Preprocess each extracted text data from 2. using the following script.

#### Usage
```
$ python src/main_lapt.py -h
usage: main_lapt.py [-h] --data_path DATA_PATH --output_data_path
                    OUTPUT_DATA_PATH [--datasets_cache_dir DATASETS_CACHE_DIR]
                    --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                    [--tokenizer_cache_dir TOKENIZER_CACHE_DIR]
                    [--num_workers NUM_WORKERS] [--max_length MAX_LENGTH]

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the input data file
  --output_data_path OUTPUT_DATA_PATH
                        Path to the output data file
  --datasets_cache_dir DATASETS_CACHE_DIR
                        Directory to cache datasets
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                        Name or path of the tokenizer to use
  --tokenizer_cache_dir TOKENIZER_CACHE_DIR
                        Directory to cache tokenizers
  --num_workers NUM_WORKERS
                        Number of worker processes to use
  --max_length MAX_LENGTH
                        Maximum length of the tokenized sequences
```

#### Example
The following preprocesses datasets for German high-resource settings.

```bash
lang_code=de
i=20
vocab_size=50000

# CVA models
python main_lapt.py \
    --data_path /path/to/${lang_code}_2^${i}_extracted.txt \
    --output_data_path /path/to/dir/cc100_${lang_code}_2^${i} \
    --datasets_cache_dir /path/to/datasets/cache/dir \
    --tokenizer_name_or_path /path/to/${lang_code}_2^${i}/ \
    --tokenizer_cache_dir /path/to/hub/cache/dir \
    --num_workers 4 \
    --max_length 2048

# LAPT
python main_lapt.py \
    --data_path /path/to/${lang_code}_2^${i}_extracted.txt \
    --output_data_path /path/to/dir/cc100_${lang_code}_2^${i}_llama2 \
    --datasets_cache_dir /path/to/datasets/cache/dir \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_cache_dir /path/to/hub/cache/dir \
    --num_workers 4 \
    --max_length 2048
```

### 5. Training a fastText model
To use FOCUS, you need to train a fastText model for each language and setting.

#### Usage
```bash
$ python src/train_fasttext.py -h
usage: Train a fasttext model with the CC-100 corpus. [-h]
                                                      --tokenizer_name_or_path
                                                      TOKENIZER_NAME_OR_PATH
                                                      [--tokenizer_cache_dir TOKENIZER_CACHE_DIR]
                                                      --text_path TEXT_PATH
                                                      [--data_cache_dir DATA_CACHE_DIR]
                                                      [--min_length MIN_LENGTH]
                                                      --target_lang
                                                      {ja,de,sw,ar,th,hi,el}
                                                      --model_cache_dir
                                                      MODEL_CACHE_DIR --suffix
                                                      {2^10,2^11,2^12,2^13,2^14,2^15,2^16,2^17,2^18,2^19,2^20,2^17_500,2^17_1000,2^17_5000,2^17_10000,2^18_500,2^18_1000,2^18_5000,2^18_10000,2^19_500,2^19_1000,2^19_5000,2^19_10000}
                                                      [--restart_from_cache]

options:
  -h, --help            show this help message and exit
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                        The tokenizer name or path.
  --tokenizer_cache_dir TOKENIZER_CACHE_DIR
                        The tokenizer cache directory.
  --text_path TEXT_PATH
                        The path to the text file.
  --data_cache_dir DATA_CACHE_DIR
                        The cache directory.
  --min_length MIN_LENGTH
                        The minimum length.
  --target_lang {ja,de,sw,ar,th,hi,el}
                        The target language.
  --model_cache_dir MODEL_CACHE_DIR
                        The model cache directory.
  --suffix {2^10,2^11,2^12,2^13,2^14,2^15,2^16,2^17,2^18,2^19,2^20,2^17_500,2^17_1000,2^17_5000,2^17_10000,2^18_500,2^18_1000,2^18_5000,2^18_10000,2^19_500,2^19_1000,2^19_5000,2^19_10000}
                        The suffix name.
  --restart_from_cache  Whether to restart from the cache.
```

#### Example
The following will train a German high-resource fastText model and save it under model_cache_dir.

```bash
#!/bin/bash

export TRANSFORMERS_CACHE=/path/to/hub/cache/dir

lang_code=de
i=20
vocab_size=50000

input_file="/path/to/${lang_code}_2^${i}_extracted.txt"

python src/train_fasttext.py \
    --tokenizer_name_or_path /path/to/${lang_code}_2^${i}_${vocab_size}/ \
    --tokenizer_cache_dir /path/to/hub/cache/dir \
    --text_path ${input_file} \
    --data_cache_dir /path/to/datasets/cache/dir \
    --min_length 5 \
    --target_lang ${lang_code} \
    --model_cache_dir /path/to/models/cache/dir \
    --suffix 2^${i}
```