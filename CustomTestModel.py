from deepeval.models import DeepEvalBaseLLM

from model_structures import *

SETTINGS_INFO = """
def structure_token_cost_1(query):
    model = "Qwen/Qwen2.5-0.5B-Instruct"

    layer0_1 = Proposer(model, query)
    layer0_2 = Proposer(model, query)
    layer0_3 = Proposer(model, query)
    layer0_4 = Proposer(model, query)
    layer0_5 = Proposer(model, query)
    layer0_6 = Proposer(model, query)

    layer1_1 = Aggregator(model, query, [layer0_1, layer0_2, layer0_3, layer0_4, layer0_5, layer0_6])

    token_costs = get_token_costs({
        "layer0_1": layer0_1,
        "layer0_2": layer0_2,
        "layer0_3": layer0_3,
        "layer0_4": layer0_4,
        "layer0_5": layer0_5,
        "layer0_6": layer0_6,
        "layer1_1": layer1_1,
    })

    return layer1_1.generate(), token_costs
"""
MODEL_STRUCTURE = structure_token_cost_1


class CustomTestModel(DeepEvalBaseLLM):
    def __init__(self):
        self.token_costs = "EMPTY"

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        result, self.token_costs = MODEL_STRUCTURE(prompt)
        return result

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "CustomTestModel"

    def get_token_costs(self):
        return self.token_costs
