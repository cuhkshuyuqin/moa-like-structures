from deepeval.models import DeepEvalBaseLLM

from model_structures import *

SETTINGS_INFO = """
def structure_1(query):
    model = "Qwen/Qwen2-7B-Instruct"

    proposer = Proposer(model, query)

    return proposer.generate()
"""
MODEL_STRUCTURE = structure_1


class CustomTestModel(DeepEvalBaseLLM):
    def __init__(self):
        pass

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        return MODEL_STRUCTURE(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "CustomTestModel"
