from typing import List

class BaseAgent:
    """
    Base class of all types of agents
    """

    def __init__(self, model_name, query, predecessors):
        """
        Args:
            model_name (str): Name of the LLM used by this agent
            query (str): User query
            predecessors (List[BaseAgent]): Predecessor agents of the agent
        """
        self.model_name = model_name
        self.query = query
        self.predecessors : List[BaseAgent] = predecessors
    
    def collect_predecessor_outputs(self):
        predecessor_outputs = []
        for predecessor in self.predecessors:
            predecessor_outputs.append(predecessor.generate)
        
        return predecessor_outputs

    def get_messages(self):
        pass
    
    def generate(self):
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    def __str__(self):
        return f"BaseAgent with name {self.model_name} and predecessors {self.predecessors}"

if __name__ == "__main__":
    pass
