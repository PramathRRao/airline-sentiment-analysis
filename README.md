# Airline Tweet Sentiment Analysis

An end-to-end NLP pipeline that analyzes sentiment from airline tweets, predicts customer churn risk, and serves predictions via a REST API.

## Tech Stack
- **NLP Model:** RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)
- **Churn Model:** XGBoost
- **API:** FastAPI + Uvicorn
- **Dashboard:** Streamlit
- **Containerization:** Docker
- **Experiment Tracking:** MLflow
- **Topic Modeling:** BERTopic
- **Explainability:** SHAP

## Project Structure
```
airline-sentiment-analysis/
├── app/                  # FastAPI application
│   └── main.py
├── data/                 # Dataset and models
├── notebooks/            # EDA, training, evaluation notebooks
├── src/                  # Source scripts
├── outputs/              # Charts and visualizations
├── mlflow_log.py         # MLflow experiment tracking
├── Dockerfile            # Container definition
└── requirements.txt      # Dependencies
```

## Setup & Installation
```bash
git clone https://github.com/your-username/airline-sentiment-analysis.git
cd airline-sentiment-analysis
pip install -r requirements.txt
```

## How to Run

**API (local):**
```bash
uvicorn app.main:app --workers 1
```

**API (Docker):**
```bash
docker build -t airline-sentiment-api .
docker run -p 8000:8000 airline-sentiment-api
```

**MLflow tracking:**
```bash
python mlflow_log.py
mlflow ui
```

## API Usage

**Endpoint:** `POST /predict`

**Request:**
```json
{"text": "I love this airline, great service!"}
```

**Response:**
```json
{"text": "I love this airline, great service!", "sentiment": "positive", "confidence": 0.9947}
```

## Model Performance

| Class    | F1 Score |
|----------|----------|
| Negative | 0.90     |
| Positive | 0.78     |
| Neutral  | 0.64     |

Churn recall: **0.96**

## Dataset
- Source: Kaggle Twitter US Airline Sentiment
- 14,640 tweets filtered to 12,679 after confidence thresholding
- 6 airlines: United, US Airways, American, Southwest, Delta, Virgin America