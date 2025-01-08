import torch
import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from logger_config import logger
from schema import NewsArticle, PredictionResponse
from model import Model
from config import get_config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

@app.on_event("startup")
async def startup_event():
    """
    Initialize the application and load the model on startup.
    """
    try:
        model_name = "bert-base-uncased"
        logger.info("Loading pre-trained model and tokenizer.")
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        device = config["DEVICE"]
        model_path = config["MODEL_PATH"]

        logger.info("Initializing Model wrapper with device: {} and model path: {}", device, model_path)
        app.state.model = Model(
            model_path=model_path, device=device, model=bert_model, tokenizer=tokenizer
        )
        app.state.model.load()

        logger.info("Model loaded successfully on startup.")
    except Exception as e:
        logger.exception("Error initializing model: {}", e)
        raise RuntimeError(f"Error initializing model: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict(article: NewsArticle):
    """
    Predict the political leaning of a news article.
    """
    try:
        logger.info("Received prediction request.")
        input_data = [f"{article.title} {article.content}"]

        prediction, confidence, probabilities = app.state.model.predict(input_data)

        logger.info(
            "Prediction successful: Class - {}, Confidence - {:.2f}",
            prediction,
            confidence,
        )
        return PredictionResponse(
            prediction=prediction, confidence=confidence, probabilities=probabilities
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
