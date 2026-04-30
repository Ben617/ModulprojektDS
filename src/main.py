import argparse

from src.data_loader import load_documents, save_predictions
from src.entities import extract_entities
from src.llm import MockLLM
from src.relations import extract_relations
from src.schemas import Prediction


def run(input_path: str, output_path: str) -> None:
    documents = load_documents(input_path)
    llm = MockLLM()

    predictions = []

    for document in documents:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample.json")
    parser.add_argument("--output", default="outputs/predictions.json")
    args = parser.parse_args()

    run(args.input, args.output)


if __name__ == "__main__":
    main()
