import pytest
from app.schema import NewsArticle, PredictionResponse

def test_news_article_valid():
    # Test with valid data
    article = NewsArticle(
        title="Sample Title",
        content="This is a sample content.",
        model_name="sample_model"
    )
    assert article.title == "Sample Title"
    assert article.content == "This is a sample content."
    assert article.model_name == "sample_model"

def test_news_article_invalid():
    # Test with invalid data (missing required fields)
    with pytest.raises(ValueError):
        NewsArticle(title="Missing Content")  # content and model_name are required

def test_prediction_response_valid():
    # Test with valid data
    response = PredictionResponse(
        prediction="Positive",
        confidence=0.95,
        probabilities=[0.1, 0.9]
    )
    assert response.prediction == "Positive"
    assert response.confidence == 0.95
    assert response.probabilities == [0.1, 0.9]

def test_prediction_response_invalid():
    # Test with invalid data (invalid type for confidence)
    with pytest.raises(ValueError):
        PredictionResponse(prediction="Positive", confidence="high", probabilities=[0.1, 0.9])  # confidence should be float
