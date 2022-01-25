import os
from typing import Dict
import yaml
import logging
import logging.config


STAGES = ["train", "test", "evaluate", "demo"]


logger = logging.getLogger(__name__)


def setup_conf(conf_path: str) -> Dict:
    with open(conf_path) as default:
        conf = yaml.safe_load(default)
    return conf


def setup_logging(base_conf_path: str, conf: str) -> None:
    """Setup logging via YAML if it is provided"""
    if conf:
        logging_yaml_path = os.path.join(base_conf_path, conf + ".yml")
        with open(logging_yaml_path) as config:
            logging.config.dictConfig(yaml.safe_load(config))


def setup_model_conf(base_conf_path: str, conf: str) -> Dict:
    """Setup a model's stage configurations"""
    if conf:
        yaml_path = os.path.join(base_conf_path, conf + ".yml")
        with open(yaml_path) as config:
            cfg = yaml.safe_load(config)
    return cfg


def conf_builder(conf_path: str, *args) -> Dict:
    """Building a custom configuration file"""
    base_conf_path = os.path.dirname(conf_path)

    conf = setup_conf(conf_path)

    setup_logging(base_conf_path, conf["logger"])

    arg_val = args[0]
    if arg_val:
        stage = arg_val.pop(0)
        if stage in STAGES:
            conf["stage"] = stage
            logger.info(f"Model's stage: '{stage}'")
        else:
            logger.error(f"Expected stages: {STAGES}, but got {stage}")
            raise ValueError(f"Expected stages: {STAGES}, but got {stage}")

    stage_conf = setup_model_conf(base_conf_path, conf["stage"])
    return stage_conf
