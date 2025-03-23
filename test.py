from openai import AzureOpenAI, OpenAI
import os
from vllm import LLM, SamplingParams
import sys

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


def double_source_test():
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


def vllm_test():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def vllm_api_test():
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    messages = [{"role": "user", "content": "What is your version?"}]

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct", messages=messages
    )
    response_content = response.choices[0].message.content
    print(response_content)


def triple_source_test():
    query = "What are 3 fun things to do in SF?"

    proposer1 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer2 = Proposer("azure", query)

    aggregator = Aggregator(
        "Qwen/Qwen2.5-Coder-32B-Instruct", query, [proposer1, proposer2]
    )

    print(aggregator.generate())


def multiple_vllm_source_test():
    query = "What are 3 fun things to do in SF?"

    proposer1 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer2 = Proposer("azure", query)

    aggregator = Aggregator(
        "Qwen/Qwen2.5-Coder-32B-Instruct", query, [proposer1, proposer2]
    )

    print(aggregator.generate())


def moa_structure_1_test():
    query = "What are 3 fun things to do in SF?"

    proposer1 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer2 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer3 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)

    aggregator1 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [proposer1, proposer2, proposer3]
    )
    aggregator2 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [proposer1, proposer2, proposer3]
    )
    aggregator3 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [proposer1, proposer2, proposer3]
    )

    aggregator4 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [aggregator1, aggregator2, aggregator3]
    )

    print(aggregator4.generate())


def moa_structure_2_test():
    query = "What are 3 fun things to do in SF?"

    proposer1 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer2 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer3 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer4 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)

    aggregator1 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [proposer1, proposer2, proposer3, proposer4]
    )

    proposer5 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer6 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)

    aggregator2 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [aggregator1, proposer5, proposer6]
    )

    print(aggregator2.generate())


def moa_structure_3_test():
    query = "What are 3 fun things to do in SF?"

    proposer1 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer2 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer3 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)

    aggregator1 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [proposer1, proposer2, proposer3]
    )

    proposer4 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer5 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer6 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)

    aggregator2 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [proposer4, proposer5, proposer6]
    )

    aggregator3 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [aggregator1, aggregator2]
    )

    print(aggregator3.generate())


def moa_structure_4_test():
    query = "What are 3 fun things to do in SF?"

    proposer1 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)
    proposer2 = Proposer("Qwen/Qwen2.5-1.5B-Instruct", query)

    aggregator1 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [proposer1, proposer2]
    )
    aggregator2 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [proposer1, proposer2]
    )

    aggregator3 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [aggregator1, aggregator2]
    )
    aggregator4 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [aggregator1, aggregator2]
    )

    aggregator5 = Aggregator(
        "Qwen/Qwen2.5-1.5B-Instruct", query, [aggregator3, aggregator4]
    )

    print(aggregator5.generate())


def qwen2_7b_instruct_test():
    query = "What are 3 fun things to do in SF?"

    proposer1 = Proposer("Alibaba-NLP/gte-Qwen2-7B-instruct", query)
    proposer2 = Proposer("Alibaba-NLP/gte-Qwen2-7B-instruct", query)

    aggregator = Aggregator(
        "Alibaba-NLP/gte-Qwen2-7B-instruct", query, [proposer1, proposer2]
    )

    print(aggregator.generate())


def local_api_test():
    HOST = "localhost"
    PORT = "18007"
    MODEL = "Qwen/QwQ-32B"


    openai_api_key = "EMPTY"
    openai_api_base = f"http://{HOST}:{PORT}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me a joke."},
        ],
        stream=True
    )

    for chunk in chat_response:
        content = chunk.choices[0].delta.content
        if content is not None:
            sys.stdout.write(content)
            sys.stdout.flush()


if __name__ == "__main__":
    local_api_test()
