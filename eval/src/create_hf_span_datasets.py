import pandas as pd
import datasets
from pathlib import Path


index_to_question_index = {
    0: "01",
    1: "02",
    2: "03",
    3: "04",
    4: "05",
    5: "06",
    6: "07",
    7: "08",
    8: "09",
    9: "10",
    10: "11",
    11: "12",
    12: "13"
}


def restructure_data(data: dict) -> dict:
    return {
        "id": data["id"],
        "context": data["context"],
        "question": data["question"],
        "answers": data["answers"],
    }


def load_kenswquad(
    dataset_path: str,
    max_context_len: int = 4096,
    num_shots: int = 3
) -> datasets.Dataset:
    # Load the dataset
    df = pd.read_csv(dataset_path)
    rows = [row[1].to_dict() for row in df.iterrows()]
    
    # Filter out samples with no corresponding text file
    new_rows = []
    for row in rows:
        sample_path = Path(dataset_path).parent / "qatexts" / str(row["Story_ID"] + ".txt")
        try:
            context = sample_path.read_text().strip().replace("\n", "")
            new_rows.append(row | {"context": context})
        except Exception:
            pass
    del rows
    
    # Convert to SQuAD format
    rows = []
    for row_id, row in enumerate(new_rows):
        for index in range(row["Num_QA_pairs"]):
            question = row["Q" + index_to_question_index[index]]
            answer_text = row["A" + index_to_question_index[index]]
            
            context_text_lower = row["context"].lower()
            answer_text_lower = answer_text.lower()
            start_position = context_text_lower.find(answer_text_lower)
            
            if start_position != -1 and len(row["context"].split()) * 0.75 <= max_context_len // (num_shots + 1):
                rows.append({
                    "id": str(row_id),
                    "context": row["context"],
                    "question": question,
                    "answers": {
                        "text": [answer_text],
                        "answer_start": [start_position]
                    }
                })
        
    return datasets.Dataset.from_pandas(pd.DataFrame(rows))
    

def main(args):
    if args.lang_code == "ja":
        task_name = "shunk031/JGLUE"
        subset_name = "JSQuAD"
    elif args.lang_code == "sw":
        pass
    else:
        task_name = "xquad"
        subset_name = "xquad." + args.lang_code
    if args.lang_code == "sw":    
        dataset = load_kenswquad(args.dataset_path)
    else:
        dataset = datasets.load_dataset(task_name, subset_name, cache_dir=args.cache_dir)
    
    # Randomly sample 500 samples from the dataset
    if args.lang_code == "sw":
        test_dataset = dataset.shuffle(seed=42).select(range(500))
        train_dataset = dataset.filter(lambda x: x not in test_dataset)
    elif args.lang_code == "ja":
        test_dataset = dataset["validation"].shuffle(seed=42).select(range(500))
        train_dataset = dataset["train"]
    else:
        test_dataset = dataset["validation"].shuffle(seed=42).select(range(500))
        train_dataset = dataset["validation"].filter(lambda x: x not in test_dataset)
    
    # Restructure the data
    train_data = []
    test_data = []
    for sample in train_dataset:
        data = restructure_data(sample)
        train_data.append(data)
    for sample in test_dataset:
        data = restructure_data(sample)
        test_data.append(data)
        
    # Convert into DataFrame
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

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
        "--dataset_path",
        type=str,
        help="[sw] The path to the dataset file",
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
