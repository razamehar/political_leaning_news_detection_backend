import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Config that serves all environment
GLOBAL_CONFIG = {
    "MODEL_PATH": os.getenv("MODEL_PATH"),
    "MODEL_PATH1": os.getenv("MODEL_PATH1"),
    "MODEL_PATH2": os.getenv("MODEL_PATH2"),
    "MLFLOW_URI": os.getenv("MLFLOW_URI"),
    "EXPERIMENT_NAME": os.getenv("EXPERIMENT_NAME"),
    "NEWS_API_KEY": os.getenv("NEWS_API_KEY"),
    "HOST": os.getenv("HOST"), 
}

# Environment specific config, or overwrite of GLOBAL_CONFIG
ENV_CONFIG = {
    "DEV": {
        "RELOAD": True,
        "DEBUG": True,
    },
    "PROD": {
        "RELOAD": False,
        "DEBUG": False,
    },
}


def get_config() -> dict:
    """
    Get the configuration based on the running environment.
    :return: A dictionary containing the configuration.
    """
    # Default to 'development' if PYTHON_ENV is not set or is empty
    env = os.getenv("PYTHON_ENV").strip()

    # Raise an error if the environment is invalid
    if env not in ENV_CONFIG:
        raise EnvironmentError(f"Configuration for environment '{env}' not found.")

    # Create the configuration by merging global and environment-specific settings
    config = {**GLOBAL_CONFIG, **ENV_CONFIG[env]}

    # Determine device based on availability
    device = 'cpu'
    config.update(
        {
            "ENV": env,
            'DEVICE': device,
        }
    )

    return config


# load config for import
CONFIG = get_config()

if __name__ == "__main__":
    # for debugging
    import json

    print(json.dumps(CONFIG, indent=4))
