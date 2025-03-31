from deepeval.benchmarks import TruthfulQA
import os

from CustomTestModel import CustomTestModel, SETTINGS_INFO
from utils import LOG_DIR, RESULTS_DIR, START_TIME


benchmark = TruthfulQA()
custom_test_model = CustomTestModel()

benchmark.evaluate(model=custom_test_model)

current_results_dir = os.path.join(RESULTS_DIR, "truthfulqa")
os.makedirs(current_results_dir, exist_ok=True)

benchmark.predictions.to_csv(
    os.path.join(current_results_dir, f"{START_TIME}_predictions.csv")
)
benchmark.task_scores.to_csv(
    os.path.join(current_results_dir, f"{START_TIME}_task_scores.csv")
)
with open(
    os.path.join(current_results_dir, f"{START_TIME}_overall_score.txt"), "w"
) as file:
    file.write(str(benchmark.overall_score))
with open(os.path.join(LOG_DIR, f"{START_TIME}_settings.txt"), "w") as file:
    file.write(SETTINGS_INFO)
