import joblib
import yaml
from pathlib import Path

from sklearn.linear_model import LogisticRegression
import pandas as pd 

import logging


logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('logs/errors.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: Path) -> dict:
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)

        logger.debug('parameters retrived from %s', params_path)
        return params
    
    except FileNotFoundError:
        raise

def load_data(data_path: Path):
    train_df = pd.read_csv(data_path)

    x_train = train_df.drop(columns=['sentiment'])
    y_train = train_df['sentiment']

    logger.debug('data loaded successfully')
    return (x_train, y_train)

def train_and_save_model(x_train, y_train, model_path, C, penalty, solver):
    try:
        lr = LogisticRegression(C=C, penalty=penalty, solver=solver)
        lr.fit(x_train, y_train)

        joblib.dump(lr, model_path)
        logger.debug('model trained and saved to %s', model_path)

    except Exception as e:
        logger.error('error in training and saving model %s', e)

if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent.parent
    params_path = root_dir / 'params.yaml'
    
    processed_dir = root_dir / 'data' / 'processed'
    data_path = processed_dir / 'train_bow.csv' 

    params = load_params(params_path)

    x_train, y_train = load_data(data_path)

    model_file_path = root_dir / 'models' / 'model.joblib'

    train_and_save_model(x_train, y_train, model_file_path, C=params['model_building']['C'], penalty=params['model_building']['penalty'], solver=params['model_building']['solver'])
