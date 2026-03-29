import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string 

import logging 
from pathlib import Path



logger = logging.getLogger('data_preprocessing')
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



def lemmatization(text, lemmatizer):
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

def remove_stop_words(text, stop_words):
    text = [word for word in str(text).split() if word not in stop_words]
    return ' '.join(text)

def remove_numbers(text):
    text = ''.join(char for char in text if not char.isdigit())
    return text

def lowercase(text):
    text = text.split()
    text = [word.lower() for word in text]
    return ' '.join(text)

def remove_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace(':', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        temp_df['content'] = temp_df['content'].apply(lowercase)
        logger.debug('data converted to lowercase')

        temp_df['content'] = temp_df['content'].apply(remove_urls)
        logger.debug('removed urls from the data')

        temp_df['content'] = temp_df['content'].apply(remove_punctuations)
        logger.debug('punctuations removed from the data')

        temp_df['content'] = temp_df['content'].apply(remove_numbers)
        logger.debug('numbers removed from the data')

        temp_df['content'] = temp_df['content'].apply(lambda x: remove_stop_words(x, stop_words))
        logger.debug('stop_words removed from the data')

        temp_df['content'] = temp_df['content'].apply(lambda x: lemmatization(x, lemmatizer))
        logger.debug('lemmatization applied to the data')

        return temp_df
    
    except Exception as e:
        logger.error('error in text_normalization: %s', e)
        raise 


if __name__ == '__main__':
    root_dir = Path(__file__).parent.parent.parent
    interim_dir = root_dir / 'data'/ 'interim'
    interim_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = root_dir / 'data' / 'raw'
    train_path = raw_dir / 'train.csv'
    test_path = raw_dir / 'test.csv'

    train_data = pd.read_csv(train_path)
    logger.debug('Train data loaded')
    test_data = pd.read_csv(test_path)
    logger.debug('Test data loaded')

    train_processed_data = normalize_text(train_data)
    logger.debug('Train data normalized')
    test_processed_data = normalize_text(test_data)
    logger.debug('Test data normalized')

    train_processed_data.to_csv(interim_dir / 'interim_train.csv', index=False)
    test_processed_data.to_csv(interim_dir / 'interim_test.csv', index=False)
    logger.debug('processed data saved to %s', interim_dir)
