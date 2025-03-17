from typing import List
from together import Together
import os
from loguru import logger
from openai import AzureOpenAI, OpenAI
from vllm import LLM, SamplingParams

from utils import DEBUG, TOGETHER_MODELS, AZURE_MODELS, VLLM_MODELS


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

    def collect_predecessor_outputs(self):
        predecessor_outputs = []
        for predecessor in self.predecessors:
            predecessor_outputs.append(predecessor.generate())

        return predecessor_outputs

    def get_messages(self):
        raise Exception("Can NOT get_messages from a BaseAgent")

    def generate(self):
        for model in TOGETHER_MODELS:
            if model == self.model_name:
                return self.generate_together()

        for model in AZURE_MODELS:
            if model == self.model_name:
                return self.generate_azure()

        for model in VLLM_MODELS:
            if model == self.model_name:
                return self.generate_vllm()

    def generate_together(self):
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_name,
            messages=self.get_messages(),
            temperature=self.temperature,
        )
        response_content = response.choices[0].message.content

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
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=self.get_messages(),
        )
        response_content = response.choices[0].message.content

        if DEBUG:
            logger.debug(f"{str(self)} generate_vllm:\n{response_content}")

        return response_content

    def __repr__(self):
        return f"BaseAgent with name {self.model_name} and predecessors {self.predecessors}"

    def __str__(self):
        return f"BaseAgent with name {self.model_name} and predecessors {self.predecessors}"


if __name__ == "__main__":
    pass
