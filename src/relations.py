import json

from src.schemas import Document, Entity, Relation
from src.prompts import build_relation_prompt
from src.json_utils import parse_json_array

def extract_relations(document: Document, entities: list[Entity], llm) -> list[Relation]:
    entity_text = json.dumps(
        [entity.__dict__ for entity in entities],
        ensure_ascii=False,
        indent=2,
    )

    prompt = build_relation_prompt(document.text, entity_text)

    raw_output = llm.generate(prompt)
    data = parse_json_array(raw_output)

    return [
        Relation(
            head=item["head"],
            tail=item["tail"],
            type=item["type"],
            evidence=item.get("evidence", ""),
        )
        for item in data
    ]
