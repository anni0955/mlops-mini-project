from dotenv import load_dotenv
import os

load_dotenv()

from flask import Flask, render_template, request
import mlflow
from preprocessing_utility import normalize_text
import joblib

mlflow.set_tracking_uri('https://dagshub.com/anni0955/mlops-mini-project.mlflow')

app = Flask(__name__)

model_name = 'LR_model'
model_version = 'latest'

model_uri = f'models:/{model_name}/{model_version}'
model = mlflow.pyfunc.load_model(model_uri)
vecotrizer = joblib.load('models/vectorizer.joblib', 'rb')


@app.route('/')
def home():
    return render_template('index.html', result=None)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = normalize_text(text)

    features = vecotrizer.transform([text])
    result = model.predict(features)
    
    return render_template('index.html', result = result[0])


app.run(debug=True)
