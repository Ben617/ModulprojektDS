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


class VLLM:
    def __init__(self, model_name: str):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model_name)
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
        )

    def generate(self, prompt: str) -> str:
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text


def create_llm(backend: str, model_name: str | None = None):
    if backend == "mock":
        return MockLLM()

    if backend == "vllm":
        if not model_name:
            raise ValueError("model_name is required when backend='vllm'")
        return VLLM(model_name)

    raise ValueError(f"Unknown backend: {backend}")
