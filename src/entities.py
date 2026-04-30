import json

from src.schemas import Document, Entity


from src.prompts import build_ner_prompt

def extract_entities(document: Document, llm) -> list[Entity]:
    prompt = build_ner_prompt(document.text)

    raw_output = llm.generate(prompt)
    data = json.loads(raw_output)

    return [
        Entity(
            id=item["id"],
            text=item["text"],
            type=item["type"],
        )
        for item in data
    ]
