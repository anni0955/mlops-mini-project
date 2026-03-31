from dotenv import load_dotenv
import os

load_dotenv()

from pydantic import BaseModel
import mlflow 
import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
import pandas as pd 
from preprocessing_utility import normalize_text



class InputText(BaseModel):
    text: str


mlflow.set_tracking_uri('https://dagshub.com/anni0955/mlops-mini-project.mlflow')


model_name = 'LR_model'
model_version = 'latest'
model = mlflow.sklearn.load_model(model_uri=f'models:/{model_name}/{model_version}')

root_dir = Path(__file__).parent
models_dir = root_dir / 'models'
models_dir.mkdir(exist_ok=True)


vectorizer = joblib.load(models_dir/'vectorizer.joblib', 'rb')


app = FastAPI(title='Sentiment Analysis API')

@app.get('/')
def home():
    return {'message': 'welcome to sentiment analysis API'}

@app.post('/predict')
def predict_sentiment(data: InputText):
    if not data.text:
        raise HTTPException(status_code=400, detail='No text provided')
    
    cleaned_text = normalize_text(data.text)
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0].max()

    sentiment = 'Happiness' if prediction == 1 else 'Sadness'

    return {
        'text': data.text,
        'prediction': sentiment,
        'confidence': round(probability, 3)
    }

