Language Adaptive Pre-training
===

## Usage
```bash
$ python src/main.py -h
usage: main.py [-h] --dataset_path DATASET_PATH
               [--val_dataset_path VAL_DATASET_PATH] --tokenizer_name_or_path
               TOKENIZER_NAME_OR_PATH --model_name_or_path MODEL_NAME_OR_PATH
               [--cache_dir CACHE_DIR] --model_type {llama2,mistral}
               [--tune_embeddings] [--r R] [--lora_alpha LORA_ALPHA]
               [--lora_dropout LORA_DROPOUT] [--no_lora] [--freeze_model]

Tune a language model.

options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path to the tokenized dataset.
  --val_dataset_path VAL_DATASET_PATH
                        Path to the tokenized validation dataset.
  --tokenizer_name_or_path TOKENIZER_NAME_OR_PATH
                        Path to the tokenizer.
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to the model.
  --cache_dir CACHE_DIR
                        Path to the cache directory.
  --model_type {llama2,mistral}
                        Model type.
  --tune_embeddings     Whether to tune the embeddings.
  --r R                 The r parameter for LoRA.
  --lora_alpha LORA_ALPHA
                        The alpha parameter for LoRA.
  --lora_dropout LORA_DROPOUT
                        The dropout parameter for LoLA.
  --no_lora             Whether to use LoRA.
  --freeze_model        Whether to freeze the model.
```

## Reproduction
**Example 1**: The following trains a German high-resource LLaMA2-7B model adapted with Merge.
```bash
lang_code=de
num_samples=2^20

python src/main.py \
    --dataset_path /path/to/datasets/cc100_${lang_code}_${num_samples} \
    --output_dir /path/to/output/dir/Llama-2-7b-hf-${lang_code}_${num_samples}-merge-tuned/ \
    --logging_dir /path/to/lowres-cva/lapt/logs/Llama-2-7b-hf-${lang_code}_${num_samples}-merge \
    --model_name_or_path /path/to/Llama-2-7b-hf-${lang_code}_${num_samples}-merge \
    --tokenizer_name_or_path /path/to/Llama-2-7b-hf-${lang_code}_${num_samples} \
    --model_type llama2 \
    --seed 42 \
    --evaluation_strategy no \
    --logging_steps 5 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --optim adamw_bnb_8bit \
    --report_to tensorboard \
    --do_train \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns False \
    --save_strategy epoch \
    --bf16 \
    --tf32 True \
    --gradient_checkpointing True \
    --tune_embeddings
```

**Example 2**: The following trains a German high-resource LLaMA2-7B model using LAPT only.
```bash
python main.py \
    --dataset_path /path/to/cc100_${lang_code}_${num_samples}_llama2 \
    --output_dir /path/to/output/Llama-2-7b-hf-${lang_code}_${num_samples} \
    --logging_dir /path/to/lowres-cva/lapt/logs/Llama-2-7b-hf-${lang_code}_${num_samples} \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --tokenizer_name_or_path meta-llama/Llama-2-7b-hf \
    --cache_dir /path/to/hub/cache/dir \
    --model_type llama2 \
    --seed 42 \
    --evaluation_strategy no \
    --logging_steps 5 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --prediction_loss_only \
    --overwrite_output_dir \
    --optim adamw_bnb_8bit \
    --report_to tensorboard \
    --do_train \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns False \
    --save_strategy epoch \
    --bf16 \
    --tf32 True \
    --gradient_checkpointing True
```