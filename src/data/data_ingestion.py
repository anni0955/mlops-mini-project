import numpy as np 
import pandas as pd 
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('logs/erros.log')
file_handler.setLevel('DEBUG')

formatter = logging.Formatter(fmt='%(asctime)s - %(name)s -%(levelname)s - %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)




def load_params(parmas_path: Path) -> dict:
    try:
        with open(parmas_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug('Parameter retrived from %s', parmas_path)
        return params
    
    except FileNotFoundError:
        logger.error('file not found: %s', parmas_path)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df 
    
    except Exception as e:
        logger.error('Data UrL is not working: %s', e)
        raise 


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=['tweet_id'])

        final_df = df[df['sentiment'].isin(['happiness', 'sadness'])]
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        
        logger.debug('Data preprocessing complete')
        return final_df
    
    except Exception as e:
        logger.error('Missing columns in dataframe %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: Path) -> None:
    try:
        raw_data_path = data_path / 'raw'
        raw_data_path.mkdir(parents=True, exist_ok=True)

        train_data.to_csv(raw_data_path / 'train.csv', index=False)
        test_data.to_csv(raw_data_path / 'test.csv', index=False)
        logger.debug('train and test data saved to %s', raw_data_path)

    except Exception as e:
        logger.error('Unexpected error %s', e)
        raise

if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent.parent

    params = load_params(root_dir / 'params.yaml')
    test_size = params['data_ingestion']['test_size']
    random_state = params['data_ingestion']['random_state']

    df = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')

    df = preprocess_data(df)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    data_path = root_dir / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    save_data(train_df, test_df, data_path)