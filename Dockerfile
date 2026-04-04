FROM python:3.10-slim

WORKDIR  /app

COPY flask_app/ /app/

COPY models/vectorizer.joblib /app/models/vectorizer.joblib

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
