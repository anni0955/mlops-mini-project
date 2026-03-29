import pandas as pd 
import joblib
from pathlib import Path
import logging
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

import mlflow
import dagshub 

dagshub.init(repo_owner='anni0955', repo_name='mlops-mini-project', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/anni0955/mlops-mini-project.mlflow')


logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('logs/erros.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_model(model_path: Path):
    try:
        with open(model_path, 'rb') as f:
            model = joblib.load(f)

        logger.debug('Model loaded from %s', model_path)
        return model
    
    except Exception as e:
        logger.error('model nor found %s', e)
        raise


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        logger.debug('data loaded from %s', data_path)

        return df

    except Exception as e:
        logger.error('path not found', e)


def evaluate_model(model, test_df: pd.DataFrame) -> dict:
    try:
        x_test = test_df.drop(columns=['sentiment'])
        y_test = test_df['sentiment']

        y_pred = model.predict(x_test)
        y_pred_prob = model.predict_proba(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob[:,1])

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
        }

        logger.debug('model evaluation metrics calculated')
        return metrics
    
    except Exception as e:
        logger.error('error in the evaluation model function %s', e)
        raise

def save_metrics(metrics, file_path: Path) -> None:
    try:
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug('metrics saved to %s', file_path)

    except Exception as e:
        logger.error('error in the saving the metrics %s', e)



if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent.parent
    models_dir = root_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / 'model.joblib'

    processed_dir = root_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    test_data_path = processed_dir / 'test_bow.csv'

    reports_dir = root_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / 'metrics.json'
    
    mlflow.set_experiment('dvc pipeline')
    with mlflow.start_run() as run:
        try:
            model = load_model(model_path)
            test_data = load_data(test_data_path)

            metrics = evaluate_model(model, test_data)
            save_metrics(metrics, metrics_path)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            
            mlflow.sklearn.log_model(model, name='model', registered_model_name="LR_model")


            mlflow.log_artifact(metrics_path)
        
        except Exception as e:
            logger.error('failed in model_evaluation %s', e)
            raise


