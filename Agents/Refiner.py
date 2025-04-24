from loguru import logger

from .BaseAgent import BaseAgent
from utils import DEBUG

aggreagator_system_prompt = """You have been provided with a response from another model to the latest user query. Your task is to refine the response. It is crucial to critically evaluate the information provided in the response, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answer but should offer an accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from the model:"""


class Refiner(BaseAgent):
    """
    Agent to aggregate answers together
    """

    def __init__(self, model_name, query, predecessors, temperature=0.0):
        """
        Args:
            model_name (str): Name of the LLM used by this agent
            query (str): User query
            predecessors (List[BaseAgent]): Predecessor agents of the agent
            temperature (float)
        """
        super().__init__(model_name, query, predecessors, temperature)
        if len(predecessors) != 1:
            raise Exception(f"Refiner has {len(predecessors)} predecessors")

        if DEBUG:
            logger.debug(f"{str(self)} created")

    def get_messages(self):
        predecessor_outputs = self.collect_predecessor_outputs()
        messages = [
            {
                "role": "system",
                "content": aggreagator_system_prompt
                + "\n"
                + predecessor_outputs[0]
            },
            {"role": "user", "content": self.query},
        ]

        if DEBUG:
            logger.debug(f"{str(self)} get_messages:\n{messages}")

        return messages

    def __repr__(self):
        return f"Aggregator with name {self.model_name} and predecessors {self.predecessors}"

    def __str__(self):
        return f"Aggregator with name {self.model_name} and predecessors {self.predecessors}"


if __name__ == "__main__":
    pass
