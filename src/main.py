import argparse

from src.data_loader import load_documents, save_predictions
from src.entities import extract_entities
from src.llm import create_llm
from src.relations import extract_relations
from src.schemas import Prediction
from tqdm import tqdm
from src.evaluation import compute_metrics, relations_to_tuples
import json
from pathlib import Path

def run(input_path: str, output_path: str, backend: str, model: str | None) -> None:
    documents = load_documents(input_path)
    llm = create_llm(backend, model)

    predictions = []

    for document in tqdm(documents, desc="Processing documents"):
        entities = extract_entities(document, llm)
        relations = extract_relations(document, entities, llm)

        prediction = Prediction(
            document_id=document.id,
            text=document.text,
            entities=entities,
            relations=relations,
        )

        predictions.append(prediction.to_dict())

    save_predictions(predictions, output_path)

    # OPTIONAL EVALUATION
    gold_path = "data/gold_sample.json"

    if Path(gold_path).exists():
        gold_data = json.loads(Path(gold_path).read_text())

        pred_map = {p["document_id"]: p for p in predictions}
        gold_map = {g["document_id"]: g for g in gold_data}

        all_pred = []
        all_gold = []

        for doc_id in gold_map:
            pred_rel = relations_to_tuples(pred_map[doc_id]["relations"])
            gold_rel = relations_to_tuples(gold_map[doc_id]["relations"])

            all_pred.extend(pred_rel)
            all_gold.extend(gold_rel)

        metrics = compute_metrics(all_pred, all_gold)

        print("\nEvaluation:")
        print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample.json")
    parser.add_argument("--output", default="outputs/predictions.json")
    parser.add_argument("--backend", default="mock", choices=["mock", "vllm"])
    parser.add_argument("--model", default=None)

    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        backend=args.backend,
        model=args.model,
    )


if __name__ == "__main__":
    main()
