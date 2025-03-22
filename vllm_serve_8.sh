uv run huggingface-cli login --token $HF_TOKEN
uv run vllm serve Qwen/QwQ-32B --port 18007
