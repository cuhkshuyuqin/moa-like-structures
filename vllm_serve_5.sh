uv run huggingface-cli login --token $HF_TOKEN
uv run vllm serve Qwen/Qwen1.5-72B-Chat --port 18004 --max-model-len 1632
