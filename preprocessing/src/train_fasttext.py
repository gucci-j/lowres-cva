from pathlib import Path

import fasttext
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def filter_fn(example, min_length: int = 5, target_lang: str = "sw"):
    if target_lang != "ja":
        if min_length <= len(example["text"].split(' ')):
            return True
        return False
    else:
        if min_length <= len(example["text"]):
            return True
        return False


def main(args):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        cache_dir=args.tokenizer_cache_dir
    )
    
    if not args.restart_from_cache:
        # Load the dataset
        dataset = load_dataset(
            "text", 
            data_files=args.text_path,
            cache_dir=args.data_cache_dir,
            split="train",
        )
        print("Before filtering:", dataset)

        # Filter the dataset
        dataset = dataset.filter(
            lambda example: filter_fn(
                example,
                args.min_length,
                args.target_lang
            )
        )
        print("After filtering:", dataset)

        # Tokenize the dataset
        dataset = dataset.map(
            lambda sample: {"text": " ".join([token for token in tokenizer.tokenize(sample["text"])])},
            num_proc=4,
        )

        # Save the tokenized dataset
        cache_path = Path(args.data_cache_dir) / f"tokenized_text_{args.target_lang}_{args.suffix}.txt"
        with cache_path.open("w+", encoding="utf-8") as f:
            f.writelines((text + "\n" for text in tqdm(dataset["text"], desc="Writing data...")))
    else:
        cache_path = Path(args.data_cache_dir) / f"tokenized_text_{args.target_lang}_{args.suffix}.txt"

    # Train the FastText model
    configs = {
        "dim": 300,
        "epochs": 3 if args.target_lang == "sw" else 1,
        "min_count": 10,
    }
    fasttext_model = fasttext.train_unsupervised(
        str(cache_path),
        dim=configs["dim"],
        neg=10,
        model="cbow",
        epoch=configs["epochs"],
        thread=4,
        minCount=configs["min_count"],
    )

    # Save the FastText model
    model_path = Path(args.model_cache_dir) / f"fasttext_model_{args.target_lang}_{args.suffix}.bin"
    fasttext_model.save_model(str(model_path))

    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Train a fasttext model with the CC-100 corpus.")
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        required=True,
        help="The tokenizer name or path."
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="The tokenizer cache directory."
    )
    parser.add_argument(
        "--text_path",
        type=str,
        required=True,
        help="The path to the text file."
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=None,
        help="The cache directory."
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="The minimum length."
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        choices=["ja", "de", "sw", "ar", "th", "hi", "el"],
        help="The target language."
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        required=True,
        help="The model cache directory."
    )
    parser.add_argument(
        "--suffix",
        type=str,
        required=True,
        choices=[
            "2^10", "2^11", "2^12", "2^13", "2^14", "2^15", "2^16", "2^17", "2^18", "2^19", "2^20",
            "2^17_500", "2^17_1000", "2^17_5000", "2^17_10000",
            "2^18_500", "2^18_1000", "2^18_5000", "2^18_10000",  
            "2^19_500", "2^19_1000", "2^19_5000", "2^19_10000",
        ],
        help="The suffix name."
    )
    parser.add_argument(
        "--restart_from_cache",
        action="store_true",
        help="Whether to restart from the cache."
    )

    args = parser.parse_args()

    main(args)
