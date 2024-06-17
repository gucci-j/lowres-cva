

import json
import os
import pandas as pd
import datasets


lang_code_to_lang_name = {
    "de": "german",
    "el": "greek",
    "ja": "japanese",
    "ar": "arabic",
    "sw": "swahili",
    "th": "thai",
    "hi": "hindi", 
}

def restructure_data(data: dict, lang_code: str) -> dict:
    if lang_code == "de":
        return {
            "id": data["gem_id"],
            "url": data["url"],
            "title": data["title"],
            "summary": data["target"],
            "text": data["text"],
        }
    else:
        return {
            "id": data["id"],
            "url": data["url"],
            "title": data["title"],
            "summary": data["summary"],
            "text": data["text"],
        }


def main(args):
    if args.lang_code == "de":
        task_name = 'GEM/mlsum'
        subset_name = args.lang_code
    elif args.lang_code == "el":
        task_name = "your-hf-id/GreekSUM"
        subset_name = None
    else:
        task_name = "csebuetnlp/xlsum"
        subset_name = lang_code_to_lang_name[args.lang_code]
    dataset = datasets.load_dataset(
        task_name, subset_name, split="test",
        cache_dir=args.cache_dir
    )

    # Load the data
    test_data = []
    for index, sample in enumerate(dataset):
        data = restructure_data(sample, lang_code=args.lang_code)
        test_data.append(data)
        
    # Convert into DataFrame
    test_df = pd.DataFrame(test_data)

    # Randomly sample 500 samples from the test set
    test_df = test_df.sample(n=500, random_state=42)

    # Save the data as .csv
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
