export OPENAI_CLIENT_CONFIG_PATH=~/client_configs.yaml

uv run alpaca_eval --output_path "results/alpaca/2025-04-24-09-00-10/" --model_outputs "results/alpaca/2025-04-24-09-00-10_output.json" --reference_outputs "results/alpaca/alpaca_eval_gpt4_baseline.json"
