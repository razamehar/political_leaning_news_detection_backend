# logger_config.py
import os
import sys
from pathlib import Path
from loguru import logger

# Ensure logs directory exists
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

def setup_logger():
    """Set up and return a logger instance based on the environment."""
    ENV = os.getenv("ENV", "DEV")  # Default to DEV if ENV is not set

    # Clear default handlers
    logger.remove()

    if ENV == "DEV":
        # add log file in development
        logger.add(logs_dir/"development.log", level="DEBUG", rotation="10 MB", retention="7 days",
                   format="{time} {level} {message}")
    else:  # Production
        logger.add(logs_dir/"production.log", level="INFO", rotation="10 MB", retention="7 days",
                   format="{time} {level} {message}")

    return logger


# Create a single, globally accessible logger instance
logger = setup_logger()
