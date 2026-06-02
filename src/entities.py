import json

from src.schemas import Document, Entity
from src.json_utils import parse_json_array

from src.prompts import build_ner_prompt

def extract_entities(document: Document, llm, prompt_path:str) -> list[Entity]:
    prompt = build_ner_prompt(document.text,prompt_path)

    raw_output = llm.generate(prompt)
    data = parse_json_array(raw_output)

    return [
        Entity(
            id=item["id"],
            text=item["text"],
            type=item["type"],
        )
        for item in data
    ]
