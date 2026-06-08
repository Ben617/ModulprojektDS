import re


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_entity_map(entities):
    return {
        entity["id"]: normalize_text(entity["text"])
        for entity in entities
        if "id" in entity and "text" in entity
    }


def relations_to_tuples(relations, entities=None):
    """
    Convert relations to comparable tuples.

    If entities are provided:
        (head_text, tail_text, relation_type)

    Otherwise:
        (head_id, tail_id, relation_type)
    """
    entity_map = build_entity_map(entities) if entities else {}

    tuples = []

    for relation in relations:
        head = relation.get("head")
        tail = relation.get("tail")
        relation_type = relation.get("type")

        if not head or not tail or not relation_type:
            continue

        if entity_map:
            head = entity_map.get(head, normalize_text(head))
            tail = entity_map.get(tail, normalize_text(tail))
        else:
            head = normalize_text(head)
            tail = normalize_text(tail)

        tuples.append((head, tail, relation_type))

    return tuples


def compute_metrics(pred_relations, gold_relations):
    pred_set = set(pred_relations)
    gold_set = set(gold_relations)

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
