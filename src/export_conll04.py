import argparse
import json
from pathlib import Path

from src.conll04_loader import load_conll04


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    args = parser.parse_args()

    docs = load_conll04(args.split)

    input_data = [
        {
            "id": doc["id"],
            "text": doc["text"],
        }
        for doc in docs
    ]

    gold_data = [
        {
            "document_id": doc["id"],
            "entities": doc["entities"],
            "relations": doc["relations"],
        }
        for doc in docs
    ]

    Path("data").mkdir(exist_ok=True)

    Path(f"data/conll04_{args.split}.json").write_text(
        json.dumps(input_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    Path(f"data/conll04_{args.split}_gold.json").write_text(
        json.dumps(gold_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
