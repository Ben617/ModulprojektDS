class MockLLM:
    def generate(self, prompt: str) -> str:
        if "Extract all named entities" in prompt:
            return """
[
  {"id": "E1", "text": "Thomas", "type": "PERSON"},
  {"id": "E2", "text": "Lisa Müller", "type": "PERSON"},
  {"id": "E3", "text": "Vancouver", "type": "LOCATION"}
]
"""

        if "Predict relations" in prompt:
            return """
[
  {
    "head": "E1",
    "tail": "E2",
    "type": "MARRIED_TO",
    "evidence": "ihre Ehe seit 20 Jahren besteht"
  }
]
"""

        return "[]"
