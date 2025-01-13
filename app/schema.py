from pydantic import BaseModel


# Pydantic models for request and response
class NewsArticle(BaseModel):
    title: str
    content: str
    model_name: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: list[float]
