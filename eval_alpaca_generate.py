import datasets
from tqdm import tqdm
import traceback
import os
import copy
import json
import asyncio
from datetime import datetime

from CustomTestModel import CustomTestModel, SETTINGS_INFO
from utils import LOG_DIR, RESULTS_DIR, START_TIME

JUMP = 0
TRYING_TIMES = 5

custom_test_model = CustomTestModel()

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
eval_set_evaluated = []

async def evaluate():
    try:
        jump_count = 0
        for example in tqdm(eval_set):
            if jump_count < JUMP:
                jump_count += 1
                continue

            new_eval = copy.deepcopy(example)

            print(" before trying  ", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            for trying in range(TRYING_TIMES):
                try:
                    print("before generate", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
                    new_eval["output"] = await custom_test_model.a_generate(example["instruction"])
                    print("after  generate", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
                    new_eval["generator"] = custom_test_model.get_model_name()
                except Exception as e:
                    traceback.print_exc()
                    continue
                break
            print("after  trying  ", datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

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


asyncio.run(evaluate())
