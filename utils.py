import toml


config = toml.load('config.toml')

logging_config = config['logging']

DEBUG = logging_config['debug']
