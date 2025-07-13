from typing import List
import os
import asyncio
import aiohttp

from loguru import logger
from together import AsyncTogether
from openai import AsyncAzureOpenAI, AsyncOpenAI
from transformers import AutoTokenizer


from utils import (
    DEBUG,
    TOGETHER_MODELS,
    AZURE_MODELS,
    VLLM_MODELS,
    VLLM_HOSTS,
    VLLM_PORTS,
    OPENROUTER_MODELS,
    OPENROUTER_MODEL_PROVIDERS,
)


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
        self.messages = None
        self.response = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.generate_task = None

    async def collect_predecessor_outputs(self):
        predecessor_outputs = await asyncio.gather(*[predecessor.generate() for predecessor in self.predecessors])

        if DEBUG:
            logger.debug(f"{str(self)} collect_predecessor_outputs:\n{predecessor_outputs}")

        return predecessor_outputs

    async def get_messages(self):
        raise Exception("Can NOT get_messages from a BaseAgent")

    async def generate(self):
        if self.response is not None:
            return self.response
        
        if self.generate_task is None:
            self.generate_task = asyncio.create_task(self.generate_once())
        
        self.response = await self.generate_task
        
        return self.response

    async def generate_once(self):
        self.messages = await self.get_messages()
        self.analyze_input_tokens(self.messages)
        
        for model in TOGETHER_MODELS:
            if model == self.model_name:
                self.response = await self.generate_together()

        for model in AZURE_MODELS:
            if model == self.model_name:
                self.response = await self.generate_azure()

        for model in VLLM_MODELS:
            if model == self.model_name:
                self.response = await self.generate_vllm()

        for model in OPENROUTER_MODELS:
            if model == self.model_name:
                self.response = await self.generate_openrouter()

        self.analyze_output_tokens(self.response)

        return self.response

    async def generate_together(self):
        async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = await async_client.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            temperature=self.temperature,
        )
        response_content = response.choices[0].message.content

        if DEBUG:
            logger.debug(f"{str(self)} generate_together:\n{response_content}")

        return response_content

    async def generate_azure(self):
        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_API_KEY"),
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )
        azure_model_name = os.getenv("AZURE_MODEL_NAME")

        response = await client.chat.completions.create(
            model=azure_model_name,
            temperature=self.temperature,
            messages=self.messages,
        )
        response_content = response.choices[0].message.content

        if DEBUG:
            logger.debug(f"{str(self)} generate_azure:\n{response_content}")

        return response_content

    async def generate_vllm(self):
        api_key = "EMPTY"
        base_url = (
            f"http://{VLLM_HOSTS[self.model_name]}:{VLLM_PORTS[self.model_name]}/v1"
        )

        if DEBUG:
            logger.debug(f"vLLM Client URL: {base_url}")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        response = await client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=self.messages,
        )
        response_content = response.choices[0].message.content

        if DEBUG:
            logger.debug(f"{str(self)} generate_vllm:\n{response_content}")

        return response_content

    async def generate_openrouter(self):
        if DEBUG:
            logger.debug(f"{str(self)} generate_openrouter:\nproviders:\n{OPENROUTER_MODEL_PROVIDERS[self.model_name]}\nmessages:\n{self.messages}")

        # client = AsyncOpenAI(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=os.getenv("OPENROUTER_API_KEY"),
        # )

        # response = await client.chat.completions.create(
        #     model=self.model_name,
        #     messages=self.messages,
        #     temperature=self.temperature,
        #     # provider={
        #     #     'order': OPENROUTER_MODEL_PROVIDERS[self.model_name],
        #     # },
        # )
        # response_content = response.choices[0].message.content

        headers = {
            "Authorization": "Bearer {}".format(os.getenv("OPENROUTER_API_KEY")),
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
                "model": self.model_name,
                "messages": self.messages,
                "temperature": self.temperature,
                "provider": {
                    "order": OPENROUTER_MODEL_PROVIDERS[self.model_name],
                    "allow_fallbacks": False
                },
            }) as response:
                if response.status == 200:
                    response_json = await response.json()
                    response_content = response_json["choices"][0]["message"]["content"]
                else:
                    logger.info(f"{str(self)} generate_openrouter:\nError when calling OpenRouter")
                    exit(1)

        if DEBUG:
            logger.debug(f"{str(self)} generate_openrouter:\nresponse:\n{response_content}")

        return response_content

    def analyze_input_tokens(self, messages):
        for item in messages:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokens = tokenizer(item["content"], return_tensors='pt')
            token_count = tokens['input_ids'].shape[1]

            self.total_input_tokens += token_count
    
    def analyze_output_tokens(self, output):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokens = tokenizer(output, return_tensors='pt')
        token_count = tokens['input_ids'].shape[1]

        self.total_output_tokens += token_count
    
    def get_total_input_tokens(self):
        return self.total_input_tokens

    def get_total_output_tokens(self):
        return self.total_output_tokens

    def __repr__(self):
        return f"BaseAgent with name {self.model_name} and predecessors {self.predecessors}"

    def __str__(self):
        return f"BaseAgent with name {self.model_name} and predecessors {self.predecessors}"


if __name__ == "__main__":
    pass
