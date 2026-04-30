from pathlib import Path


PROMPT_DIR = Path("prompts")


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8")


def build_ner_prompt(text: str) -> str:
    template = load_prompt("ner_prompt.txt")
    return template.format(text=text)


def build_relation_prompt(text: str, entities_json: str) -> str:
    template = load_prompt("relation_prompt.txt")
    return template.format(text=text, entities=entities_json)
