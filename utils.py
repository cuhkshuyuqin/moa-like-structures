import toml
from loguru import logger
from datetime import datetime
import os


START_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

config = toml.load("config.toml")
logging_config = config["logging"]
inference_config = config["inference"]

DEBUG = logging_config["debug"]

TOGETHER_MODELS = inference_config["together"]
AZURE_MODELS = inference_config["azure"]
VLLM_MODELS = inference_config["vllm"]
VLLM_HOSTS = inference_config["vllm_hosts"]
VLLM_PORTS = inference_config["vllm_ports"]

LOG_DIR = "logs"

logger.remove(0)
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, f"{START_TIME}.log"))

RESULTS_DIR = "results"
