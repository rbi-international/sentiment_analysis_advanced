import logging
import os
from datetime import datetime

def get_logger(name: str = "sentiment_logger", log_file: str = None, log_level: str = "INFO"):
    """
    Creates and returns a logger object with both console and file handlers.
    
    Args:
        name (str): Name of the logger
        log_file (str): Path to log file. If None, a default file in 'logs/' will be used.
        log_level (str): Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate logs if logger is reused
    if logger.handlers:
        return logger

    # Default log file path with timestamp
    if not log_file:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/log_{timestamp}.log"

    # Formatter
    formatter = logging.Formatter(
        fmt="🔍 [%(asctime)s] [%(levelname)s] - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("✅ Logger initialized")

    return logger
