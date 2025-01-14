from logger_config import logger
import torch
import numpy as np
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig
from config import get_config

class Model:
    """
    Wrapper for PyTorch model loading and inference.
    """

    def __init__(self, model_path: str, model, tokenizer):
        self.model_path = model_path
        self.model = model
        self.tokenizer = tokenizer
        logger.info("Initialized Model instance with model path: {}", model_path)

    def load(self, device, model_config=None):
        """Load the model state dictionary."""
        try:
            logger.info(f"Loading model from {self.model_path} using {device}")
            if model_config is not None:
                logger.info("Applying model configuration.")
                self.model = get_peft_model(self.model, model_config)
            self.model.load_state_dict(torch.load(self.model_path, map_location=device, weights_only=True))
            self.model.to(device)
            self.model.eval()
            logger.info("Model loaded successfully from {}", self.model_path)
        except FileNotFoundError:
            logger.error("Model file not found: {}", self.model_path)
            raise
        except Exception as e:
            logger.exception("Failed to load model.")
            raise

    def predict(self, input_data, device):
        """Perform predictions."""
        try:
            logger.info("Starting prediction for input data")
            # Tokenize input data
            encodings = self.tokenizer(
                input_data, truncation=True, padding=True, max_length=128, return_tensors="pt"
            )
            encodings = {key: val.to(device) for key, val in encodings.items()}
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)
                confidence, prediction_index = torch.max(probabilities, dim=1)
            # Map prediction index to class
            classes = ["Left", "Center", "Right"]
            prediction = classes[prediction_index.item()]
            prediction_values = probabilities.cpu().numpy().flatten().tolist()

            logger.info("Prediction completed: Class - {}, Confidence - {:.2f}", prediction, confidence.item())
            return prediction, confidence.item(), prediction_values

        except Exception as e:
            logger.exception("Error during prediction.")
            raise
