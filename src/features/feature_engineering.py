import pandas as pd 
import logging 
from pathlib import Path 

from sklearn.feature_extraction.text import CountVectorizer

import yaml
import joblib

logger = logging.getLogger('feature_engineering')
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

def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
        df = df.fillna('')
        logger.debug('data loaded and missing imputed from %s', data_path)

        return df
    
    except FileNotFoundError:
        logger.error('file is not found at the path')
        raise

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int, save_path: Path):
    try:
        vectorizer = CountVectorizer(max_features=max_features)

        x_train = train_data['content']
        y_train = train_data['sentiment']

        x_test = test_data['content']
        y_test = test_data['sentiment']

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        feature_names = vectorizer.get_feature_names_out()

        train_df = pd.DataFrame(x_train_bow.toarray(), columns=feature_names)
        train_df['sentiment'] = y_train.values
        test_df = pd.DataFrame(x_test_bow.toarray(), columns=feature_names)
        test_df['sentiment'] = y_test.values
        
        joblib.dump(vectorizer, save_path)
        logger.debug('vectorizer model saved to %s', save_path)
        return (train_df, test_df)

    except Exception as e:
        logger.error('Problem in Bag of words %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: Path):
    try:
        train_data.to_csv(data_path / 'train_bow.csv', index=False)
        test_data.to_csv(data_path / 'test_bow.csv', index=False)
        logger.debug('Data saved to %s', data_path)

    except Exception as e:
        logger.error('path does not exist')
        raise


if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent.parent
    processed_dir = root_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    interim_dir = root_dir / 'data' / 'interim'
    interim_dir.mkdir(parents=True, exist_ok=True)

    params_path = root_dir / 'params.yaml'
    params = load_params(params_path)

    max_features = params['feature_engineering']['max_features']

    train_data_path = interim_dir / 'interim_train.csv'
    train_data = load_data(train_data_path)

    test_data_path = interim_dir / 'interim_test.csv'
    test_data = load_data(test_data_path)

    models_dir = root_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_save_path = models_dir / 'vectorizer.joblib'
    train_df, test_df = apply_bow(train_data, test_data, max_features, vectorizer_save_path)

    save_data(train_df, test_df, processed_dir)

