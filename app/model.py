from logger_config import logger
import torch
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
        logger.info("Initialized Model instance with device: {} and model path: {}", device, model_path)

    def load(self):
        """Load the model state dictionary."""
        try:
            logger.info("Loading model from: {}", self.model_path)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully from {}", self.model_path)
        except FileNotFoundError:
            logger.error("Model file not found: {}", self.model_path)
            raise
        except Exception as e:
            logger.exception("Failed to load model.")
            raise

    def predict(self, input_data):
        """Perform predictions."""
        try:
            logger.info("Starting prediction for input data")
            
            # Tokenize input data
            encodings = self.tokenizer(
                input_data, truncation=True, padding=True, max_length=128, return_tensors="pt"
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
            prediction_values = probabilities.cpu().numpy().flatten().tolist()

            logger.info("Prediction completed: Class - {}, Confidence - {:.2f}", prediction, confidence.item())
            return prediction, confidence.item(), prediction_values

        except Exception as e:
            logger.exception("Error during prediction.")
            raise
