import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string 

from pathlib import Path

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def lemmatization(text):

    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

def remove_stop_words(text):
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

def normalize_text(text: str) -> pd.DataFrame:
    text = lowercase(text)
    text = remove_urls(text)
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_stop_words(text)
    text = lemmatization(text)
    return text 
