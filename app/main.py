import torch
import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Get the path to the current file
current_file = Path(__file__).resolve()

# Get the parent directory (i.e., backend)
parent_dir = current_file.parent

# Append the 'app' directory to sys.path
sys.path.append(str(parent_dir))

from logger_config import logger
from schema import NewsArticle, PredictionResponse
from model import Model
from config import get_config
from peft import LoraConfig
from constants import NEWS_SOURCES
from news_app import get_outlet_news

# Load the configuration
config = get_config()
logger.info("Starting FastAPI app...")

# Initialize FastAPI app
app = FastAPI(
    title="Political Leaning News Detection API",
    description="An API for predicting political leaning of news articles and evaluating ML models using MLflow.",
    version="1.0",
)

# Allow CORS for local debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}

@app.on_event("startup")
async def startup_event():
    """
    Initialize the application and load the model on startup.
    """
    base_model_name = "bert-base-uncased"
    model_paths = {
        "model": config["MODEL_PATH"],
        "quantized_model": config["MODEL_PATH1"],
        "lora_model": config["MODEL_PATH2"]
    }
    logger.info("Loading pre-trained BERT model and tokenizer.")
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    for model_name, model_path in model_paths.items():
        try:
            model_config = None
            device = config["DEVICE"]
            
            if model_name == "lora_model":
                model_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    bias="none"
                )
            logger.info(f"Initializing Model wrapper for {model_name} with device: {device} and model path: {model_path}")
            model_wrapper = Model(model_path=model_path, model=base_model, tokenizer=tokenizer)
            model_wrapper.load(device=device, model_config=model_config)  # Load the fine-tuned weights
            
            # Store the loaded models
            models[model_name] = (model_wrapper, device, model_config)
            logger.info(f"{model_name} loaded successfully on startup.")

        except Exception as e:
            logger.exception(f"Failed to load {model_name}: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict(article: NewsArticle):
    """
    Predict the political leaning of a news article based on the selected fine-tuned model.
    """
    try:
        logger.info("Received prediction request.")
        # Get the selected fine-tuned model from the request
        selected_model = article.model_name

        if selected_model not in models:
            raise HTTPException(status_code=400, detail="Invalid model name selected.")
        
        input_data = [f"{article.title} {article.content}"]
        device = models[selected_model][1]
        # Use the selected fine-tuned model for prediction
        logger.info("Performing prediction using model: {}", selected_model)
        prediction, confidence, probabilities = models[selected_model][0].predict(input_data, device=models[selected_model][1])

        logger.info(
            "Prediction successful: Class - {}, Confidence - {:.2f}",
            prediction,
            confidence,
        )
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities
        )
    except Exception as e:
        logger.exception("Prediction failed: {}", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/outlets")
def get_news_outlet():
    """
    Retrieve a list of available news outlets.
    """
    try:
        logger.info("Fetching list of news outlets.")
        sources = NEWS_SOURCES.keys()
        return list(sources)
    except Exception as e:
        logger.exception("Failed to fetch news outlets: {}", e)
        raise HTTPException(status_code=500, detail="Failed to fetch news outlets.")

@app.get("/outlets/{outlet}")
def get_news_details(outlet: str):
    """
    Retrieve details of a specific news outlet.
    """
    try:
        logger.info("Fetching details for outlet: {}", outlet)
        if outlet in NEWS_SOURCES:
            outlet = NEWS_SOURCES[outlet]
        else:
            logger.warning("No outlets found for: {}", outlet)
            raise HTTPException(status_code=404, detail="No outlets found.")
        response = get_outlet_news(outlet)
        logger.info("Successfully fetched news for outlet: {}", outlet)
        return response
    except Exception as e:
        logger.exception("Failed to fetch news details for outlet {}: {}", outlet, e)
        raise HTTPException(status_code=500, detail="Failed to fetch news details.")

if __name__ == "__main__":
    logger.info("Starting application server.")
    uvicorn.run(
        "main:app",
        host=config.get("HOST", "0.0.0.0"),
        port=config.get("PORT", 8000),
        reload=config.get("RELOAD", False),
        log_level="debug" if config.get("DEBUG", False) else "info",
    )
