import json
import re


def extract_json_array(text: str) -> str:
    """
    Extract first JSON array from a string.
    """
    match = re.search(r"\[\s*{.*?}\s*]", text, re.DOTALL)
    if match:
        return match.group(0)
    return "[]"


def parse_json_array(text: str):
    """
    Robust JSON parsing for LLM outputs.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = extract_json_array(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return []
