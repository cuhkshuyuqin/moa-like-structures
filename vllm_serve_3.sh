uv run huggingface-cli login --token $HF_TOKEN
uv run vllm serve google/gemma-3-1b-it --port 8002
