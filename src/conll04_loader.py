from datasets import load_dataset


ENTITY_TYPE_MAP = {
    "Peop": "PERSON",
    "Loc": "LOCATION",
    "Org": "ORGANIZATION",
    "Other": "OTHER",
}


def entity_text(tokens, start, end):
    return " ".join(tokens[start:end])


def load_conll04(split="train"):
    dataset = load_dataset("DFKI-SLT/conll04")[split]

    documents = []

    for i, item in enumerate(dataset):
        tokens = item["tokens"]
        text = " ".join(tokens)

        entities = []
        for entity_index, entity in enumerate(item["entities"]):
            entities.append({
                "id": f"E{entity_index + 1}",
                "text": entity_text(tokens, entity["start"], entity["end"]),
                "type": ENTITY_TYPE_MAP.get(entity["type"], entity["type"]),
            })

        relations = []
        for relation in item["relations"]:
            relations.append({
                "head": f"E{relation['head'] + 1}",
                "tail": f"E{relation['tail'] + 1}",
                "type": relation["type"],
            })

        documents.append({
            "id": f"conll04_{split}_{i}",
            "text": text,
            "entities": entities,
            "relations": relations,
            "orig_id": item.get("orig_id"),
        })

    return documents
