from Agents.Proposer import Proposer
from Agents.Aggregator import Aggregator
from utils import TOGETHER_MODELS, AZURE_MODELS


def basic_moa_test():
    query = "What are 3 fun things to do in SF?"

    proposer1 = Proposer("microsoft/WizardLM-2-8x22B", query)
    proposer2 = Proposer("microsoft/WizardLM-2-8x22B", query)
    proposer3 = Proposer("microsoft/WizardLM-2-8x22B", query)

    aggregator = Aggregator(
        "microsoft/WizardLM-2-8x22B", query, [proposer1, proposer2, proposer3]
    )

    print(aggregator.generate())


def inference_config_test():
    print(TOGETHER_MODELS)
    print(AZURE_MODELS)


def multiple_source_test():
    query = "What are 3 fun things to do in SF?"

    proposer1 = Proposer("microsoft/WizardLM-2-8x22B", query)
    proposer2 = Proposer("gpt-4o/2024-02-15-preview", query)

    aggregator = Aggregator(
        "Qwen/Qwen2.5-Coder-32B-Instruct", query, [proposer1, proposer2]
    )

    print(aggregator.generate())


if __name__ == "__main__":
    multiple_source_test()
