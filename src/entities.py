from src.json_utils import parse_json_array
from src.prompts import build_ner_prompt
from src.schemas import Document, Entity


def extract_entities(document: Document, llm, prompt_path: str) -> list[Entity]:
    prompt = build_ner_prompt(document.text, prompt_path)
    raw_output = llm.generate(prompt)
    data = parse_json_array(raw_output)

    entities = []

    for i, item in enumerate(data):
        if "text" not in item or "type" not in item:
            continue

        entities.append(
            Entity(
                id=item.get("id", f"E{i + 1}"),
                text=item["text"],
                type=item["type"],
            )
        )

    return entities
