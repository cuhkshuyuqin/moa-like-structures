import datasets
from tqdm import tqdm
import traceback
import os
import copy
import json

from CustomTestModel import CustomTestModel, SETTINGS_INFO
from utils import LOG_DIR, RESULTS_DIR, START_TIME

JUMP = 0
TRYING_TIMES = 5

custom_test_model = CustomTestModel()

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
eval_set_evaluated = []
try:
    jump_count = 0
    for example in tqdm(eval_set):
        if jump_count < JUMP:
            jump_count += 1
            continue

        new_eval = copy.deepcopy(example)

        for trying in range(TRYING_TIMES):
            try:
                new_eval["output"] = custom_test_model.generate(example["instruction"])
                new_eval["generator"] = custom_test_model.get_model_name()
            except Exception as e:
                traceback.print_exc()
                continue
            break

        eval_set_evaluated.append(new_eval)

        # break # if only process 1 test case

except BaseException as e:
    traceback.print_exc()

current_results_dir = os.path.join(RESULTS_DIR, "alpaca")
os.makedirs(current_results_dir, exist_ok=True)

with open(os.path.join(current_results_dir, f"{START_TIME}_settings.txt"), "w") as file:
    file.write(SETTINGS_INFO)

with open(os.path.join(current_results_dir, f"{START_TIME}_output.json"), "w", encoding="utf-8") as json_file:
    json.dump(eval_set_evaluated, json_file, ensure_ascii=False, indent=4)

with open(os.path.join(current_results_dir, f"{START_TIME}_token_costs.txt"), "w") as file:
    file.write(custom_test_model.get_token_costs())
