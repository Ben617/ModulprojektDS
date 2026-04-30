def compute_metrics(pred_relations, gold_relations):
    """
    pred_relations / gold_relations: list of tuples (head, tail, type)
    """

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


def relations_to_tuples(relations):
    return [(r["head"], r["tail"], r["type"]) for r in relations]
