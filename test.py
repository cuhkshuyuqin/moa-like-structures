from openai import AzureOpenAI
import os

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
    proposer2 = Proposer("azure", query)

    aggregator = Aggregator(
        "Qwen/Qwen2.5-Coder-32B-Instruct", query, [proposer1, proposer2]
    )

    print(aggregator.generate())


def azure_version_test():
    messages = [{"role": "user", "content": "What is your version?"}]

    llm = AzureOpenAI(
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    )
    azure_model_name = os.getenv("AZURE_MODEL_NAME")

    response = llm.chat.completions.create(
        model=azure_model_name,
        temperature=0,
        messages=messages,
    )
    response_content = response.choices[0].message.content
    print(response_content)


if __name__ == "__main__":
    multiple_source_test()
