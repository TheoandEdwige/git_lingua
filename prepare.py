import re
import nltk
import unicodedata
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from collections import defaultdict


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
    # Initialize a tokenizer object
    tokenizer = ToktokTokenizer()
    # Tokenize the input data using the tokenizer object
    string = tokenizer.tokenize(string, return_str=True)
    string = re.sub(r"[^a-z0-9\s]", "", string)
    string = re.sub(r"\s\d{1}\s", "", string)
    # Return the processed data
    return string

# Function to remove stopwords
def remove_stopwords(string, exclude_words=None):
    extra_words = ['ai', 'artificial intelligence', 'machine learning', 'deep learning']
    exclude_words = exclude_words or []
    stopword_list = stopwords.words('english')
    stopword_list = set(stopword_list) - set(exclude_words)
    words = string.split()
    words = [w for w in words if w not in stopword_list]
    words = [w for w in words if w not in extra_words]
    return ' '.join(words)

# Function to perform text preprocessing on a DataFrame
def preprocess_text_in_dataframe(dataframe, column_name, exclude_words=None):
    extra_words = ['ai', 'artificial', 'intelligence', 'machinelearning', 'deep learning']
    # Basic cleaning
    dataframe[column_name] = dataframe[column_name].apply(basic_clean)
    
    # Tokenizationg
    dataframe[column_name] = dataframe[column_name].apply(tokenize)
    
    # Removing stopwords
    dataframe[column_name] = dataframe[column_name].apply(remove_stopwords, extra_words, exclude_words=exclude_words)

    return dataframe

def big_and_small(df) -> dict:
    '''
    creates a dictionary that measuresin  how many documnets a word appears that is ordered by count.
    can use the dictionary to drop a lot of words that are not descriptive.
    '''
    for readme in df['readme'].dropna():
        # Tokenize the 'Readme' content
        words = set(readme.split()) # Using set to ensure unique

    # Initialize a defaultdict of set to track in which readmes each word has appeared
    word_readmes = defaultdict(set)

    # Iterate through each 'Readme' entry with its index
    for indx, readme in df['readme'].dropna().items():
        # Tokenize the 'Readme' content
        words = set(readme.split()) # Using set to ensure unique
        for word in words:
            word_readmes[word].add(indx)

    # Count in how many different readmes each word appears
    word_counts = {word: len(readmes) for word, readmes in word_readmes.items()}
    
    
    sorted_words = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
    return sorted_words



    

