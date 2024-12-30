import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Config that serves all environment
GLOBAL_CONFIG = {
    "MODEL_PATH": os.getenv("MODEL_PATH", "model/model.pt"),
    "MLFLOW_URI": os.getenv("MLFLOW_URI", "http://localhost:5000"),
    "EXPERIMENT_NAME": os.getenv("EXPERIMENT_NAME", "Political News Detection"),
    "DEVICE": "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu"),
    "HOST": os.getenv("HOST", ""),  # Check for host configuration
}

# Environment specific config, or overwrite of GLOBAL_CONFIG
ENV_CONFIG = {
    "development": {
        "RELOAD": True,
        "DEBUG": True,
    },
    "staging": {"RELOAD": False, "DEBUG": True},
    "production": {
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
    env = os.getenv("PYTHON_ENV", "development").strip()

    # Raise an error if the environment is invalid
    if env not in ENV_CONFIG:
        raise EnvironmentError(f"Configuration for environment '{env}' not found.")

    # Create the configuration by merging global and environment-specific settings
    config = {**GLOBAL_CONFIG, **ENV_CONFIG[env]}

    # # Determine device based on availability
    # if torch.cuda.is_available() and config.get('USE_CUDA_IF_AVAILABLE', False):
    #     device = 'cuda'
    # elif torch.backends.mps.is_available() and config.get('USE_MPS_IF_AVAILABLE', False):
    #     device = 'mps'
    # else:
    #     device = 'cpu'
    config.update(
        {
            "ENV": env,
            # 'DEVICE': device,
        }
    )

    return config


# load config for import
CONFIG = get_config()

if __name__ == "__main__":
    # for debugging
    import json

    print(json.dumps(CONFIG, indent=4))
