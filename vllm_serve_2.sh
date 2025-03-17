uv run huggingface-cli login --token $HF_TOKEN
uv run vllm serve meta-llama/Llama-3.2-1B --port 8001
