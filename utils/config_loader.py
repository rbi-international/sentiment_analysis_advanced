# utils/config_loader.py

import yaml
import os
from pathlib import Path
from logutils.logger import get_logger

logger = get_logger("config_loader")

def load_config(config_path: str = None) -> dict:
    """
    Load YAML configuration file.
    
    Args:
        config_path (str): Optional custom path. If None, will auto-detect project root.
    
    Returns:
        dict: Dictionary containing configuration values.
    """
    try:
        # 🔒 Automatically resolve the absolute path to config.yaml
        if config_path is None:
            root = Path(__file__).resolve().parents[1]  # sentiment_analysis_advanced/
            config_path = os.path.join(root, "config", "config.yaml")

        logger.debug(f"🔍 Looking for config at: {config_path}")

        if not os.path.exists(config_path):
            logger.error(f"❌ Config file not found at path: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"✅ Loaded config from {config_path}")
            return config

    except yaml.YAMLError as e:
        logger.exception(f"🔥 YAML parsing error: {e}")
        raise
    except Exception as e:
        logger.exception(f"🔥 Unexpected error loading config: {e}")
        raise
