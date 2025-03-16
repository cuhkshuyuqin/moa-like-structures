from Agents.Proposer import Proposer
from Agents.Aggregator import Aggregator

query = "What are 3 fun things to do in SF?"

proposer1 = Proposer("microsoft/WizardLM-2-8x22B", query)
proposer2 = Proposer("microsoft/WizardLM-2-8x22B", query)
proposer3 = Proposer("microsoft/WizardLM-2-8x22B", query)

aggregator = Aggregator(
    "microsoft/WizardLM-2-8x22B", query, [proposer1, proposer2, proposer3]
)

print(aggregator.generate())
