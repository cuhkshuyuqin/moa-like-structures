uv run huggingface-cli login --token $HF_TOKEN
uv run vllm serve meta-llama/Llama-3.1-8B-Instruct --port 18005
