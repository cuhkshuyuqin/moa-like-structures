from loguru import logger

from .BaseAgent import BaseAgent
from utils import DEBUG


class Proposer(BaseAgent):
    """
    Agent with no predecessors
    """

    def __init__(self, model_name, query, temperature=0.7):
        """
        Args:
            model_name (str): Name of the LLM used by this agent
            query (str): User query
            predecessors (List[BaseAgent]): Predecessor agents of the agent
            temperature (float)
        """
        super().__init__(model_name, query, [], temperature)

        if DEBUG:
            logger.debug(f"{str(self)} created")

    async def get_messages(self):
        messages = [{"role": "user", "content": self.query}]

        if DEBUG:
            logger.debug(f"{str(self)} get_messages:\n{messages}")

        return messages

    def __repr__(self):
        return (
            f"Proposer with name {self.model_name} and predecessors {self.predecessors}"
        )

    def __str__(self):
        return (
            f"Proposer with name {self.model_name} and predecessors {self.predecessors}"
        )


if __name__ == "__main__":
    pass
