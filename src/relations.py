import json

from src.json_utils import parse_json_array
from src.prompts import build_relation_prompt
from src.schemas import Document, Entity, Relation


ALLOWED_RELATIONS = {
    "Located_In",
    "Work_For",
    "OrgBased_In",
    "Live_In",
    "Kill",
}


def extract_relations(
    document: Document,
    entities: list[Entity],
    llm,
    prompt_path: str,
) -> list[Relation]:
    entities_json = json.dumps(
        [entity.__dict__ for entity in entities],
        ensure_ascii=False,
        indent=2,
    )

    prompt = build_relation_prompt(document.text, entities_json, prompt_path)
    raw_output = llm.generate(prompt)
    data = parse_json_array(raw_output)

    valid_entity_ids = {entity.id for entity in entities}
    relations = []

    for item in data:
        head = item.get("head")
        tail = item.get("tail")
        relation_type = item.get("type")

        if relation_type not in ALLOWED_RELATIONS:
            continue

        if head not in valid_entity_ids:
            continue

        if tail not in valid_entity_ids:
            continue

        if head == tail:
            continue

        relations.append(
            Relation(
                head=head,
                tail=tail,
                type=relation_type,
                evidence=item.get("evidence", ""),
            )
        )

    return relations
