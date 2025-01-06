import torch
import uvicorn
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from schema import NewsArticle, PredictionResponse
from model import Model
from config import get_config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from constants import NEWS_SOURCES
from news_app import get_outlet_news


# Load the configuration
config = get_config()
logger.add("logs/app.log", rotation="1 MB", level="DEBUG")
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
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=3
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        device = config["DEVICE"]
        model_path = config["MODEL_PATH"]

        app.state.model = Model(
            model_path=model_path, device=device, model=bert_model, tokenizer=tokenizer
        )
        app.state.model.load()

        logger.info("Model loaded successfully on startup.")
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        raise RuntimeError(f"Error initializing model: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
def predict(article: NewsArticle):
    """
    Predict the political leaning of a news article.
    """
    try:
        input_data = [f"{article.title} {article.content}"]
        prediction, confidence, probabilities = app.state.model.predict(input_data)
        return PredictionResponse(
            prediction=prediction, confidence=confidence, probabilities=probabilities
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/outlets")
def get_news_outlet():
    sources = NEWS_SOURCES.keys()
    return list(sources)


@app.get("/outlets/{outlet}")
def get_news_details(outlet: str):
    if outlet in NEWS_SOURCES:
        outlet = NEWS_SOURCES[outlet]
    else:
        raise Exception("No outlets found")
    response = get_outlet_news(outlet)
    return response


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.get("HOST", "0.0.0.0"),
        port=config.get("PORT", 8000),
        reload=config.get("RELOAD", False),
        log_level="debug" if config.get("DEBUG", False) else "info",
    )
