from deepeval.models import DeepEvalBaseLLM

from model_structures import *

SETTINGS_INFO = """
def qwen3_30b_a3b_test(query):
    model = "qwen/qwen3-30b-a3b"

    proposer = Proposer(model, query)

    return proposer.generate()
"""
MODEL_STRUCTURE = qwen3_30b_a3b_test


class CustomTestModel(DeepEvalBaseLLM):
    def __init__(self):
        self.token_costs_input = None
        self.token_costs_output = None

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        result, token_costs = MODEL_STRUCTURE(prompt)
        self.sum_token_costs(token_costs)
        return result

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "CustomTestModel"

    def get_token_costs(self):
        token_costs = ""
        total_input_tokens_count = 0
        total_output_tokens_count = 0
        for name, instance in self.token_costs_input.items():
            token_costs += f"{name}:\ninput:  {self.token_costs_input[name]}\noutput: {self.token_costs_output[name]}\ntotal:  {self.token_costs_input[name] + self.token_costs_output[name]}\n\n"
            total_input_tokens_count += self.token_costs_input[name]
            total_output_tokens_count += self.token_costs_output[name]
        
        token_costs += f"sum:\ninput:  {total_input_tokens_count}\noutput: {total_output_tokens_count}\ntotal:  {total_input_tokens_count + total_output_tokens_count}"

        return token_costs

    def sum_token_costs(self, token_costs):
        if self.token_costs_input is None:
            self.token_costs_input = {}
            self.token_costs_output = {}
            for name, instance in token_costs.items():
                self.token_costs_input[name] = instance.get_total_input_tokens()
                self.token_costs_output[name] = instance.get_total_output_tokens()
        else:
            for name, instance in token_costs.items():
                self.token_costs_input[name] += instance.get_total_input_tokens()
                self.token_costs_output[name] += instance.get_total_output_tokens()
