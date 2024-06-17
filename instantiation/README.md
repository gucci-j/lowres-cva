Target Model Initialization
===

This is to initialize embeddings and LM heads using various cross-lingual vocabulary adaptation techniques listed in Section 3.

## Usage
```bash
$ python src/main.py -h
usage: Initialize the target model. [-h] --source_model_name_or_path
                                    SOURCE_MODEL_NAME_OR_PATH
                                    --target_tokenizer_name_or_path
                                    TARGET_TOKENIZER_NAME_OR_PATH --output_dir
                                    OUTPUT_DIR [--cache_dir CACHE_DIR]
                                    [--proj]
                                    [--method {merge,align,random,mean,focus}]
                                    [--dataset_path DATASET_PATH]
                                    [--use_only_merge_for_head]
                                    [--use_only_merge_for_embeddings]
                                    [--use_only_align] [--consider_mean]
                                    [--fasttext_model_path FASTTEXT_MODEL_PATH]

options:
  -h, --help            show this help message and exit
  --source_model_name_or_path SOURCE_MODEL_NAME_OR_PATH
                        The source model to initialize the target model with.
  --target_tokenizer_name_or_path TARGET_TOKENIZER_NAME_OR_PATH
                        The target tokenizer to initialize the target model
                        with.
  --output_dir OUTPUT_DIR
                        The output directory to save the target model and
                        tokenizer.
  --cache_dir CACHE_DIR
                        The cache directory to save the source model and
                        tokenizer.
  --proj                Whether to apply the output projection init.
  --method {merge,align,random,mean,focus}
                        The method to initialize the target model.
  --dataset_path DATASET_PATH
                        [align] The path to the dataset for aligning the
                        target tokenizer.
  --use_only_merge_for_head
                        [align] Whether to use only merge-based init for an LM
                        head.
  --use_only_merge_for_embeddings
                        [align] Whether to use only merge-based init for
                        embeddings.
  --use_only_align      [align] Whether to use only align-based init.
  --consider_mean       [align] Whether to consider the mean of embeddings for
                        align-based init.
  --fasttext_model_path FASTTEXT_MODEL_PATH
                        [focus] The path to the FastText model.
```

## Reproduction
**Example 1**:  The following is to initialize a LLaMA2-7B model for German using Random in high-resource settings.
```bash
lang_code=de
i=20

# random
python src/main.py \
    --source_model_name_or_path meta-llama/Llama-2-7b-hf \
    --target_tokenizer_name_or_path /path/to/${lang_code}_2^${i} \
    --output_dir /path/to/output/dir/Llama-2-7b-hf-${lang_code}_2^${i}-rand \
    --cache_dir /path/to/hub/cache/dir \
    --method random \
    --proj
```

**Example 2**: The following is to initialize a LLaMA2-7B model for German using FOCUS in high-resource settings.
```bash
lang_code=de
i=20

# focus
python src/main.py \
    --source_model_name_or_path meta-llama/Llama-2-7b-hf \
    --target_tokenizer_name_or_path /path/to/${lang_code}_2^${i} \
    --output_dir /path/to/output/dir/Llama-2-7b-hf-${lang_code}_2^${i}-focus \
    --cache_dir /path/to/hub/cache/dir \
    --method focus \
    --proj
    --fasttext_model_path /path/to/models/cache/dir/fasttext_model_${lang_code}_2^${i}.bin
```
