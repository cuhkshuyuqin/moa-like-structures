from typing import List
from together import Together
import os
from loguru import logger

from utils import DEBUG


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
        pass

    def generate(self):
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_name,
            messages=self.get_messages(),
            temperature=self.temperature,
        )
        response_content = response.choices[0].message.content

        if DEBUG:
            logger.debug(f"{str(self)} get_messages:\n{response_content}")

        return response_content

    def __repr__(self):
        return f"BaseAgent with name {self.model_name} and predecessors {self.predecessors}"

    def __str__(self):
        return f"BaseAgent with name {self.model_name} and predecessors {self.predecessors}"


if __name__ == "__main__":
    pass
