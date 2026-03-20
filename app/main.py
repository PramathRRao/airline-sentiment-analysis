from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
app = FastAPI()

# Load model
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data","models", "airline_model","airline_model")
print("Model path:", model_path)
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