import json
from pathlib import Path

from src.schemas import Document


def load_documents(path: str) -> list[Document]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    documents = []
    for item in data:
        documents.append(
            Document(
                id=str(item["id"]),
                text=str(item["text"]),
            )
        )

    return documents


def save_predictions(predictions: list[dict], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
