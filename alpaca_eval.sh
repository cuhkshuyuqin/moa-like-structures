export OPENAI_CLIENT_CONFIG_PATH=~/client_configs.yaml

uv run alpaca_eval --model_outputs "results/alpaca/2025-04-05-15-23-12_output_combined.json" --reference_outputs "results/alpaca/alpaca_eval_gpt4_baseline.json"
