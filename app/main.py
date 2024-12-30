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

# Load the configuration
CONFIG = get_config()

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

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        model_path = CONFIG["MODEL_PATH"]

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


@app.get("/model/metrics")
def get_model_metrics():
    """
    Retrieve model evaluation metrics logged in MLflow.
    """
    try:
        mlflow.set_tracking_uri(CONFIG["MLFLOW_URI"])
        mlflow.set_experiment(CONFIG["EXPERIMENT_NAME"])

        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(CONFIG["EXPERIMENT_NAME"])
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")

        runs = client.search_runs(
            experiment.experiment_id,
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise HTTPException(status_code=404, detail="No runs found in experiment")

        latest_run = runs[0]
        metrics = latest_run.data.metrics

        return metrics
    except Exception as e:
        logger.error(f"Failed to retrieve model metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve model metrics: {str(e)}"
        )


@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API status.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=CONFIG.get("HOST", "0.0.0.0"),
        port=CONFIG.get("PORT", 8000),
        reload=CONFIG.get("RELOAD", False),
        log_level="debug" if CONFIG.get("DEBUG", False) else "info",
    )
