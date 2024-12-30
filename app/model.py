import torch
from loguru import logger
import numpy as np
import torch.nn.functional as F


class Model:
    """
    Wrapper for PyTorch model loading and inference.
    """

    def __init__(self, model_path: str, device: str, model, tokenizer):
        self.device = device
        self.model_path = model_path
        self.model = model
        self.tokenizer = tokenizer

    def load(self):
        """Load the model state dictionary."""
        try:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully from {self.model_path}.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def predict(self, input_data):
        """Perform predictions."""
        try:
            # Tokenize input data
            encodings = self.tokenizer(
                input_data,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            encodings = {key: val.to(self.device) for key, val in encodings.items()}

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=1)
                confidence, prediction_index = torch.max(probabilities, dim=1)

            # Map prediction index to class
            classes = ["Left", "Center", "Right"]
            prediction = classes[prediction_index.item()]
            prediction_values = probabilities.cpu().numpy()
            rounded_prob = np.round(prediction_values, 2)
            rounded_prob_list = rounded_prob.flatten().tolist()
            return prediction, confidence.item(), rounded_prob_list
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Error during prediction: {e}")
