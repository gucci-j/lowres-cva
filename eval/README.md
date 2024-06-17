Evaluation
===

This is to evaluate models using LightEval.

## Reproduction
### 1. Preprocessing
First, you need to generate HF datasets for evaluation.

#### NLI
```bash
#!/bin/bash

cd ./src

lang_codes=(
    "ja"
    "de"
    "ar"
    "sw"
    "el"
    "th"
    "hi"
)

for lang_code in "${lang_codes[@]}"; do
    python create_hf_nli_datasets.py \
        --output_dir /path/to/temp/output/dir \
        --cache_dir /path/to/hub/cache/dir \
        --repo_id /your-hf-id/nli-${lang_code} \
        --lang_code ${lang_code}
done
```

#### SPAN
You need to download the KenSwQuAD dataset from its corresponding websites. We cannot redistribute them due to the license agreement.  
* KenSwQuAD - https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OTL0LM

Then, run the following:
```bash
#!/bin/bash

cd ./src

lang_codes=(
    "ja"
    "de"
    "ar"
    "sw"
    "el"
    "th"
    "hi"
)

for lang_code in "${lang_codes[@]}"; do
    python create_hf_span_datasets.py \
        --output_dir /path/to/temp/output/dir \
        --cache_dir /path/to/hub/cache/dir \
        --repo_id /your-hf-id/span-${lang_code} \
        --dataset_path /path/to/kenswquad/KenSwQuAD_final_7526_QA_pairs_csv.csv \
        --lang_code ${lang_code}
done
```

#### SUM
You need to generate GreekSUM dataset at first. Please follow the procedures instructed [here](https://github.com/iakovosevdaimon/GreekSUM).

Then, you need to convert generated CSV files into HF datasets using our conversion script:
```bash
$ python src/convert_greeksum_into_hf_datasets.py
usage: convert_greeksum_into_hf_datasets.py [-h] [--data_dir DATA_DIR]
                                            [--split_train_data_path SPLIT_TRAIN_DATA_PATH]
                                            [--split_test_data_path SPLIT_TEST_DATA_PATH]
                                            [--train_data_path TRAIN_DATA_PATH]
                                            [--test_data_path TEST_DATA_PATH]
                                            [--repo_name REPO_NAME]
                                            [--repo_dir REPO_DIR]

Create HF datasets

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to the json data directory
  --split_train_data_path SPLIT_TRAIN_DATA_PATH
                        Path to the split train data
  --split_test_data_path SPLIT_TEST_DATA_PATH
                        Path to the split test data
  --train_data_path TRAIN_DATA_PATH
                        Path to save the train data
  --test_data_path TEST_DATA_PATH
                        Path to save the test data
  --repo_name REPO_NAME
                        The name of the repository to which the data will be
                        uploaded
  --repo_dir REPO_DIR   The path to the repository directory
```

Finally, run the following:
```bash
#!/bin/bash
cd ./src/

lang_codes=(
    "ja"
    "de"
    "ar"
    "sw"
    "el"
    "th"
    "hi"
)

for lang_code in "${lang_codes[@]}"; do
    python create_hf_sum_datasets.py \
        --output_dir /path/to/temp/output/dir \
        --cache_dir /path/to/hub/cache/dir \
        --repo_id /your-hf-id/sum-${lang_code} \
        --lang_code ${lang_code}
done
```

### 2. Modify paths
You need to modify all the dataset ids in the following scripts:
* [nli.py](./src/nli.py)
* [sum.py](./src/sum.py)
* [span.py](./src/span.py)

### 3. Generate config files
You need to generate a config file for each adapted model. You could use our handy script:
```bash
$ python src/generate_yml.py -h
usage: generate_yml.py [-h] [--adapter_path ADAPTER_PATH]
                       [--base_path BASE_PATH] [--output_path OUTPUT_PATH]

Generate a yml file

options:
  -h, --help            show this help message and exit
  --adapter_path ADAPTER_PATH
  --base_path BASE_PATH
  --output_path OUTPUT_PATH
```

### 4. Run evaluation
**For Source:**
```bash
lang_code=de
model_path=/model/path/or/name
custom_task_script_dir=/path/to/extreme-cva/eval/src

cd /path/to/lighteval

tasks=(
    "custom|nli:${lang_code}|0|0"
    "custom|nli:${lang_code}|5|0"
    "custom|span:${lang_code}|0|0"
    "custom|span:${lang_code}|3|1"
    "custom|sum:${lang_code}|0|1"
)

for task in "${tasks[@]}"; do
    # get a task name with out lang_code
    task_name=$(echo $task | cut -d'|' -f2 | cut -d':' -f1)

    accelerate launch --mixed_precision=bf16 run_evals_accelerate.py \
        --model_args "pretrained=${model_path}" \
        --tasks "${task}" \
        --custom_tasks "${custom_task_script_dir}/${task_name}.py" \
        --override_batch_size 1 \
        --output_dir="${log_dir}/${task_name}" \
        --cache_dir="${cache_dir}"
done
```

**For adapted models**
```bash
lang_code=de
yml_dir=/path/to/config/dir
custom_task_script_dir=/path/to/extreme-cva/eval/src

cd /path/to/lighteval

tasks=(
    "custom|nli:${lang_code}|0|0"
    "custom|nli:${lang_code}|5|0"
    "custom|span:${lang_code}|0|0"
    "custom|span:${lang_code}|3|1"
    "custom|sum:${lang_code}|0|1"
)

for task in "${tasks[@]}"; do
    # get a task name with out lang_code
    task_name=$(echo $task | cut -d'|' -f2 | cut -d':' -f1)

    accelerate launch --mixed_precision=bf16 run_evals_accelerate.py \
       --model_config_path ${yml_dir}/${model_name}.yml \
        --tasks "${task}" \
        --custom_tasks "${custom_task_script_dir}/${task_name}.py" \
        --override_batch_size 1 \
        --output_dir="${log_dir}/${task_name}" \
        --cache_dir="${cache_dir}"
done
```

## Reference
* https://github.com/huggingface/lighteval
