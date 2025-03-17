import toml


config = toml.load("config.toml")
logging_config = config["logging"]
inference_config = config["inference"]

DEBUG = logging_config["debug"]

TOGETHER_MODELS = inference_config["together"]
AZURE_MODELS = inference_config["azure"]
VLLM_MODELS = inference_config["vllm"]
