import unittest
import mlflow

import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path

import joblib
from dotenv import load_dotenv


class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root_dir = Path(__file__).parent.parent
        dotenv_path = root_dir / '.env'

        load_dotenv(dotenv_path)

        username = os.getenv('MLFLOW_TRACKING_USERNAME')
        password = os.getenv('MLFLOW_TRACKING_PASSWORD')

        if not username or not password:
            raise EnvironmentError('MLFLOW dagshub credential not found in .env')
        
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password


        mlflow.set_tracking_uri('https://dagshub.com/anni0955/mlops-mini-project.mlflow')

        cls.model_name = 'LR_model'
        cls.model_version = 'latest'
        cls.model = mlflow.sklearn.load_model(f'models:/{cls.model_name}/{cls.model_version}')

        vectorizer_path = root_dir / 'models' / 'vectorizer.joblib'
        test_data_path = root_dir / 'data' / 'processed' / 'test_bow.csv'

        with open(vectorizer_path, 'rb') as f:
            cls.vectorizer = joblib.load(f)

        cls.holdout_data = pd.read_csv(test_data_path)

    def test_model_loader_properly(self):
        self.assertIsNotNone(self.model)

    def test_model_signature(self):
        input_text = 'i got colorful marks in the end sem exmanination.'
        input_data = self.vectorizer.transform([input_text])
        
        input_df = pd.DataFrame(input_data.toarray(), columns=self.vectorizer.get_feature_name_out())
        prediction = self.model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_name_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)


if __name__ == '__main__':
    unittest.main()
