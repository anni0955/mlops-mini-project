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
        
        input_df = pd.DataFrame(input_data.toarray(), columns=self.vectorizer.get_feature_names_out())
        prediction = self.model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        x_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.model.predict(x_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        expected_accuracy = .75
        expected_precision = .75
        expected_recall = .75
        expected_f1 = .75

        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'accuracy should be atleast {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'precision should be atleast {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'accuracy should be atleast {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'accuracy should be atleast {expected_f1}')

if __name__ == '__main__':
    unittest.main()
