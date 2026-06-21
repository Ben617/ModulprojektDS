from src.json_utils import parse_json_array
from src.prompts import build_ner_prompt
from src.schemas import Document, Entity


ALLOWED_ENTITY_TYPES = {
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "OTHER",
}


def extract_entities(document: Document, llm, prompt_path: str) -> list[Entity]:
    prompt = build_ner_prompt(document.text, prompt_path)
    raw_output = llm.generate(prompt)
    return parse_entities(raw_output)

def parse_entities(raw_output: str) -> list[Entity]:
    data = parse_json_array(raw_output)
    entities = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        entity_type = item.get("type")

        if not text or not entity_type:
            continue

        if entity_type not in ALLOWED_ENTITY_TYPES:
            continue

        entities.append(
            Entity(
                id=item.get("id", f"E{i + 1}"),
                text=text,
                type=entity_type,
            )
        )

    return entities