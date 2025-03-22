uv run huggingface-cli login --token $HF_TOKEN
uv run vllm serve alpindale/WizardLM-2-8x22B --port 18003
