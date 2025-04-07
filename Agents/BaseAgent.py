from typing import List
from together import Together
import os
from loguru import logger
from openai import AzureOpenAI, OpenAI
from transformers import AutoTokenizer

from utils import (
    DEBUG,
    TOGETHER_MODELS,
    AZURE_MODELS,
    VLLM_MODELS,
    VLLM_HOSTS,
    VLLM_PORTS,
)


class BaseAgent:
    """
    Base class of all types of agents
    """

    def __init__(self, model_name, query, predecessors, temperature):
        """
        Args:
            model_name (str): Name of the LLM used by this agent
            query (str): User query
            predecessors (List[BaseAgent]): Predecessor agents of the agent
            temperature (float)
        """
        self.model_name = model_name
        self.query = query
        self.predecessors: List[BaseAgent] = predecessors
        self.temperature = temperature
        self.response = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def collect_predecessor_outputs(self):
        predecessor_outputs = []
        for predecessor in self.predecessors:
            predecessor_outputs.append(predecessor.generate())

        return predecessor_outputs

    def get_messages(self):
        raise Exception("Can NOT get_messages from a BaseAgent")

    def generate(self):
        if self.response is not None:
            return self.response

        for model in TOGETHER_MODELS:
            if model == self.model_name:
                self.response = self.generate_together()
                return self.response

        for model in AZURE_MODELS:
            if model == self.model_name:
                self.response = self.generate_azure()
                return self.response

        for model in VLLM_MODELS:
            if model == self.model_name:
                self.response = self.generate_vllm()
                return self.response

    def generate_together(self):
        messages = self.get_messages()
        self.analyze_input_tokens(messages)

        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        response_content = response.choices[0].message.content
        self.analyze_output_tokens(response_content)

        if DEBUG:
            logger.debug(f"{str(self)} generate_together:\n{response_content}")

        return response_content

    def generate_azure(self):
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        )
        azure_model_name = os.getenv("AZURE_MODEL_NAME")

        response = client.chat.completions.create(
            model=azure_model_name,
            temperature=self.temperature,
            messages=self.get_messages(),
        )
        response_content = response.choices[0].message.content

        if DEBUG:
            logger.debug(f"{str(self)} generate_azure:\n{response_content}")

        return response_content

    def generate_vllm(self):
        messages = self.get_messages()
        self.analyze_input_tokens(messages)

        api_key = "EMPTY"
        base_url = (
            f"http://{VLLM_HOSTS[self.model_name]}:{VLLM_PORTS[self.model_name]}/v1"
        )

        if DEBUG:
            logger.debug(f"vLLM Client URL: {base_url}")

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=messages,
        )
        response_content = response.choices[0].message.content
        self.analyze_output_tokens(response_content)

        if DEBUG:
            logger.debug(f"{str(self)} generate_vllm:\n{response_content}")

        return response_content

    def analyze_input_tokens(self, messages):
        for item in messages:
            tokenizer = AutoTokenizer.from_pretrained(self.name)
            tokens = tokenizer(item["content"], return_tensors='pt')
            token_count = tokens['input_ids'].shape[1]

            self.total_input_tokens += token_count
    
    def analyze_output_tokens(self, output):
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        tokens = tokenizer(output, return_tensors='pt')
        token_count = tokens['input_ids'].shape[1]

        self.total_output_tokens += token_count
    
    def get_total_input_tokens(self):
        return total_input_tokens

    def get_total_output_tokens(self):
        return total_output_tokens

    def __repr__(self):
        return f"BaseAgent with name {self.model_name} and predecessors {self.predecessors}"

    def __str__(self):
        return f"BaseAgent with name {self.model_name} and predecessors {self.predecessors}"


if __name__ == "__main__":
    pass
