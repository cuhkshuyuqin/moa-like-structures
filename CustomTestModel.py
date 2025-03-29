from deepeval.models import DeepEvalBaseLLM

from model_structures import *
from utils import START_TIME

SETTINGS_INFO = "Simple Qwen/Qwen2-7B-Instruct"
MODEL_STRUCTURE = structure_2


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
