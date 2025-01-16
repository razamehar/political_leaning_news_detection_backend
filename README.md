# Political Leaning Detection in News Headlines
This project aims to analyze political bias in mainstream media by classifying news articles from various outlets as right-leaning, centrist, or left-leaning. The goal is to assess potential biases within these news sources and provide insights into their political orientations. The implementation leverages transformers for sequence classification, alongside tools for preprocessing, training, and evaluating machine learning models.

## Possible Use Cases
### Helping people understand news bias
- Classifies news articles to show if they lean left, right, or center, and 
- Aims to help readers make informed choices by identifying bias in news sources

### Supporting NGOs and journalists with media monitoring
- Monitors bias trends on specific issues, helping ensure balanced reporting
- Useful for detecting potential misinformation or slanted perspectives

### Assisting academic research
- Useful for researchers studying media bias, journalism, or political influence in the news

## The Dataset
Released on 15 July 2020, the POLUSA dataset contains 0.9 million political news articles, carefully balanced across different periods and news outlet popularity. It provides a valuable resource for analyzing political trends and biases in media. The dataset is available for download on Zenodo.org.

Link: https://zenodo.org/records/3946057/files/polusa_balanced.zip?download=1

## Setup
Follow these steps to set up the project:

### Clone the repository
```bash
git clone https://github.com/iampujan/political_leaning_news_detection_backend.git
cd your-repository
```

### Install UV Python package and project manager

#### For macOS and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
#### For Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install dependencies
```bash
uv sync
```

### Usage
Run the notebook sequentially in a Jupyter Notebook environment or a similar setup:

- Step 1: Download data using gdown or any alternative method.
- Step 2: Preprocess text data with tokenization, stopword removal, and bigram extraction.
- Step 3: Fine-tune a transformers model using the provided pipeline.
- Step 4: Evaluate model performance and generate classification reports.

### Reproducibility
To reproduce the results:

1. Ensure the dependencies are installed as described in the Setup section.
2. Follow the cell execution sequence in the notebook:
   - Data download and exploration.
   - Data preprocessing.
   - Model training and evaluation.
3. Save results and logs using mlflow.

## Running the Application
### Backend (FastAPI)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The server should start on:

Development: http://localhost:8080
Production: https://plndapp.gentleground-f2e94450.italynorth.azurecontainerapps.io/

### Frontend (React)
```bash
npm start
```
The server should start on http://localhost:3000 

## Contact
For any questions or clarifications, please contact Raza Mehar at [raza.mehar@gmail.com], Pujan Thapa at [iampujan@outlook.com] or Syed Najam Mehdi at [najam.electrical.ned@gmail.com].

