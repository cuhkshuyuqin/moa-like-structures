from BaseAgent import BaseAgent

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
        BaseAgent.__init__(model_name, query, [])
        self.temperature = temperature

    def get_messages(self):
        return [{"role": "user", "content": self.query}]

    def __str__(self):
        return f"Proposer with name {self.model_name} and predecessors {self.predecessors}"

if __name__ == "__main__":
    pass
