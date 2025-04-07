import os

from utils import (
    VLLM_MODELS,
    VLLM_PORTS,
)

# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME = "meta-llama/Llama-3.2-1B"
# MODEL_NAME = "google/gemma-3-1b-it"

# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
# MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
# MODEL_NAME = "Qwen/Qwen2-Math-7B-Instruct"

# MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"

# MODEL_NAME = "Qwen/QwQ-32B"

# MODEL_NAME = "alpindale/WizardLM-2-8x22B"
# MODEL_NAME = "Qwen/Qwen1.5-72B-Chat"

MODEL_NAME = "wzhouad/gemma-2-9b-it-WPO-HB"

MAX_MODEL_LEN = None
# MAX_MODEL_LEN = 16384

if MODEL_NAME not in VLLM_MODELS:
    raise Exception("model not configured")

port = VLLM_PORTS[MODEL_NAME]
huggingface_token = os.getenv("HF_TOKEN")

instruction_login = f"uv run huggingface-cli login --token {huggingface_token}"
instruction_serve = f"uv run vllm serve {MODEL_NAME} --port {port} --trust-remote-code --gpu-memory-utilization 1"
if MAX_MODEL_LEN is not None:
    instruction_serve += f" --max-model-len {MAX_MODEL_LEN}"

os.system(instruction_login + " && " + instruction_serve)
