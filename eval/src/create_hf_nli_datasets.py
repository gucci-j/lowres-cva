

import json
import os
import pandas as pd
import datasets


jnli_label_to_xnli_label = {
    0: 0, # entailment
    1: 2, # contradiction
    2: 1  # neutral
}

def restructure_data(data: dict, lang_code: str, sample_id: int = None) -> dict:
    if lang_code == "ja":
        return {
            "id": data["sentence_pair_id"],
            "premise": data["sentence1"],
            "hypothesis": data["sentence2"],
            "label": jnli_label_to_xnli_label[data["label"]],
        }
    else:
        return {
            "id": sample_id,
            "premise": data["premise"],
            "hypothesis": data["hypothesis"],
            "label": data["label"],
        }


def main(args):
    if args.lang_code == "ja":
        task_name = "shunk031/JGLUE"
        subset_name = "JNLI"
    else:
        task_name = "xnli"
        subset_name = args.lang_code
    dataset = datasets.load_dataset(task_name, subset_name, cache_dir=args.cache_dir)

    # Load the data
    train_data = []
    test_data = []
    if args.lang_code == "ja":
        for sample in dataset["train"]:
            data = restructure_data(sample, lang_code=args.lang_code)
            train_data.append(data)
    else:
        for index, sample in enumerate(dataset["train"]):
            data = restructure_data(sample, lang_code=args.lang_code, sample_id=index)
            train_data.append(data)
    if args.lang_code == "ja":
        for sample in dataset["validation"]:
            data = restructure_data(sample, lang_code=args.lang_code)
            test_data.append(data)
    else:
        for index, sample in enumerate(dataset["test"]):
            data = restructure_data(sample, lang_code=args.lang_code, sample_id=index)
            test_data.append(data)
        
    # Convert into DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    # Randomly sample 500 samples from the test set
    test_df = test_df.sample(n=500, random_state=42)

    # Save the data as .csv
    train_df.to_json(args.output_dir + "/train.jsonl", lines=True, orient="records", force_ascii=False)
    test_df.to_json(args.output_dir + "/test.jsonl", lines=True, orient="records", force_ascii=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create HF datasets')
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="The directory to save the downloaded files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The directory to save the output files",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="The name of the repository to which the data will be uploaded",
    )
    parser.add_argument(
        "--lang_code",
        type=str,
        help="The language code of the dataset",
    )
    args = parser.parse_args()
    main(args)


    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.create_repo(
            repo_id=args.repo_id, 
            private=True,
            repo_type='dataset',
        )
    except Exception:
        pass
    api.upload_folder(
        folder_path=args.output_dir,
        repo_id=args.repo_id,
        repo_type='dataset',
    )
