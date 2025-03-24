from deepeval.benchmarks import GSM8K
import os

from CustomTestModel import CustomTestModel

benchmark = GSM8K()
custom_test_model = CustomTestModel()

benchmark.evaluate(model=custom_test_model)

benchmark.predictions.to_csv(os.path.join("results", "gsm8k", f"{START_TIME}_predictions.csv"))
benchmark.task_scores.to_csv(os.path.join("results", "gsm8k", f"{START_TIME}_task_scores.csv"))
with open(os.path.join("results", "gsm8k", f"{START_TIME}_overall_score.txt"), 'w') as file:
    file.write(str(benchmark.overall_score))
with open(os.path.join("logs", f"{START_TIME}_settings.txt"), 'w') as file:
    file.write(SETTINGS_INFO)
