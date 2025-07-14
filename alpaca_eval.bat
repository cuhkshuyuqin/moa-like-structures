@REM set OPENAI_API_BASE="https://openrouter.ai/api/v1"
@REM set OPENAI_API_KEY="..."

uv run alpaca_eval --output_path "results/alpaca/2025-07-13-20-56-41/" --model_outputs "results/alpaca/2025-07-13-20-56-41_output_combined.json" --reference_outputs "results/alpaca/alpaca_eval_gpt4_baseline.json"
