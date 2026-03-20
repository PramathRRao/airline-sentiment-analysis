from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import boto3
import tempfile

app = FastAPI()

# Download model from S3
s3 = boto3.client('s3')
bucket = "airline-sentiment-model-bucket"
model_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]

model_dir = tempfile.mkdtemp()
for file in model_files:
    s3.download_file(bucket, f"airline_model/{file}", os.path.join(model_dir, file))

model_path = model_dir
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Input schema
class TweetInput(BaseModel):
    text: str

# Labels
labels = {0: "negative", 1: "neutral", 2: "positive"}

@app.get("/")
def home():
    return {"message": "Airline Sentiment API is running"}

@app.post("/predict")
def predict(tweet: TweetInput):
    inputs = tokenizer(
        tweet.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1).max().item()
    
    return {
        "text": tweet.text,
        "sentiment": labels[predicted_class],
        "confidence": round(confidence, 4)
    }