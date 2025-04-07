import datasets
from tqdm import tqdm
import traceback
import os
import copy
import json

from CustomTestModel import CustomTestModel, SETTINGS_INFO
from utils import LOG_DIR, RESULTS_DIR, START_TIME

TRYING_TIMES = 5
FILE_PATH = "results/alpaca/2025-04-05-15-23-12_output_combined.json"

custom_test_model = CustomTestModel()

eval_set = []
with open(FILE_PATH, 'r', encoding='utf-8') as json_file:
    eval_set = json.load(json_file)

try:
    for example in tqdm(eval_set):
        if example["generator"] != custom_test_model.get_model_name():
            for trying in range(TRYING_TIMES):
                try:
                    example["output"] = custom_test_model.generate(example["instruction"])
                except Exception as e:
                    continue
                break
except BaseException as e:
    traceback.print_exc()

with open(FILE_PATH, "w", encoding="utf-8") as json_file:
    json.dump(eval_set, json_file, ensure_ascii=False, indent=4)
