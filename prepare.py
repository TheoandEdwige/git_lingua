import re
import nltk
import unicodedata
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer


# Function for basic text cleaning
def basic_clean(string):
    string = string.lower()
    string = unicodedata.normalize('NFKD', string)\
            .encode('ascii', 'ignore')\
            .decode('utf-8', 'ignore')
    string = re.sub(r"[^a-z0-9\s]", "", string)
    string = string.strip()
    return string

# Function to tokenize text
def tokenize(string):
    tokenizer = ToktokTokenizer()
    string = tokenizer.tokenize(string, return_str=True)
    return string

# Function to remove stopwords
def remove_stopwords(string, extra_words=None, exclude_words=None):
    extra_words = extra_words or []
    exclude_words = exclude_words or []
    stopword_list = stopwords.words('english')
    stopword_list = set(stopword_list) - set(exclude_words)
    words = string.split()
    words = [w for w in words if w not in stopword_list]
    words = [w for w in words if w not in extra_words]
    return ' '.join(words)

# Function to perform text preprocessing on a DataFrame
def preprocess_text_in_dataframe(dataframe, column_name, extra_words=None, exclude_words=None):
    # Basic cleaning
    dataframe[column_name] = dataframe[column_name].apply(basic_clean)
    
    # Tokenization
    dataframe[column_name] = dataframe[column_name].apply(tokenize)
    
    # Removing stopwords
    dataframe[column_name] = dataframe[column_name].apply(remove_stopwords, extra_words=extra_words, exclude_words=exclude_words)
    
    return dataframe




