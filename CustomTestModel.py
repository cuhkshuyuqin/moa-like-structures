from deepeval.models import DeepEvalBaseLLM

from model_structures import *

SETTINGS_INFO = """
def structure_self_moa_sota(query):
    model = "wzhouad/gemma-2-9b-it-WPO-HB"

    layer0_1 = Proposer(model, query, 0.7)
    layer0_2 = Proposer(model, query, 0.7)
    layer0_3 = Proposer(model, query, 0.7)
    layer0_4 = Proposer(model, query, 0.7)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4], 0.7)

    return layer1_1.generate()
"""
MODEL_STRUCTURE = structure_self_moa_sota


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
