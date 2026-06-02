from pathlib import Path


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_ner_prompt(text: str, prompt_path: str) -> str:
    template = load_prompt(prompt_path)
    return template.format(text=text)


def build_relation_prompt(text: str, entities_json: str, prompt_path: str) -> str:
    template = load_prompt(prompt_path)
    return template.format(text=text, entities=entities_json)
