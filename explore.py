import numpy as np
import prepare as p
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_basic_statistics(dataframe, column_name):
    # Calculate basic statistics on the specified column
    basic_stats = dataframe[column_name].describe()
    return basic_stats

def identify_most_common_words(dataframe, column_name, top_n=10):
    # Tokenize the text and count word frequencies
    all_text = " ".join(dataframe[column_name])
    words = all_text.split()
    word_counts = pd.Series(words).value_counts()
    # Get the top N most common words
    top_words = word_counts.head(top_n)
    return top_words

def analyze_readme_lengths(dataframe, column_name):
    # Analyze the distribution of README lengths
    readme_lengths = dataframe[column_name].str.len()
    return readme_lengths

def generate_word_cloud(dataframe, column_name):
    # Generate a word cloud from the specified column
    all_text = " ".join(dataframe[column_name])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def plot_readme_length_histogram(df):
    readme_lengths = df['readme'].str.len()
    
    plt.figure(figsize=(10, 5))
    plt.hist(readme_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('README Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of README Lengths')
    plt.show()

#visualization of the top words in readme
def top_words_barplot(top_words):
    top_words.plot.barh()
    plt.show()


def idf_plot():
    ## testing IDF CHART

    n_documents = 20

    x = np.arange(1, n_documents + 1)
    y = np.log(n_documents / x)

    plt.figure(figsize=(12, 8))
    plt.plot(x, y, marker='.')

    plt.xticks(x)
    plt.xlabel('# of Documents the word appears in')
    plt.ylabel('IDF')
    plt.title('IDF for a given word')
    plt.show()


def hypothesis_one(df):
    # Tokenize the 'readme' column into words
    df['readme_words'] = df['readme'].apply(lambda x: len(x.split()))
    # Create the 'readme_word_count' column
    df['readme_word_count'] = df['readme_words']
    # EDA and Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x='language', y='readme_word_count', data=df)
    plt.title('Impact of Programming Language on README Word Count')
    plt.xticks(rotation=90)
    plt.xlabel('Programming Language')
    plt.ylabel('Average README Word Count')
    plt.show()


def hypothesis_two(df):
    # Define the top 10 programming languages
    top_10_languages = ['Python', 'JavaScript', 'Java', 'C++', 'C#', 'Ruby', 'Swift', 'PHP', 'Go', 'TypeScript']
    
    # Filter the DataFrame to include only the top 10 languages
    df_top_10 = df[df['language'].isin(top_10_languages)]
    
    # Calculate the word count for each README
    df_top_10['readme_word_count'] = df_top_10['readme'].apply(lambda x: len(x.split()))
    
    # EDA and Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='readme_word_count', y='language', data=df_top_10)
    plt.title('Relationship Between README Word Count and Top 10 Programming Languages')
    plt.xlabel('README Word Count')
    plt.ylabel('Programming Language')
    plt.xticks(rotation=90)
    plt.show()


def hypothesis_three(df):
    # Filter the DataFrame to include only R repositories
    r_df = df[df['language'] == 'R']
    
    # Preprocess the text data in the README files (use your preprocess_text_in_dataframe function)
    r_df = p.preprocess_text_in_dataframe(r_df, 'readme')
    
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Fit and transform the READMEs into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(r_df['readme'])
    
    # Get the feature names (words) corresponding to the columns of the TF-IDF matrix
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Calculate the mean TF-IDF score for each word across R READMEs
    mean_tfidf_scores = tfidf_matrix.mean(axis=0)
    
    # Convert the mean TF-IDF scores to a dictionary with words as keys and scores as values
    word_scores = {word: score for word, score in zip(feature_names, mean_tfidf_scores.tolist()[0])}
    
    # Find the top 3 most predictive words (highest TF-IDF scores)
    top_predictive_words = Counter(word_scores).most_common(3)
    
    # Display the top 3 most predictive words
    print("Top 3 Most Predictive Words in R READMEs:")
    for word, score in top_predictive_words:
        print(f"{word}: {score}")


def hypothesis_four(df):
    # Filter the DataFrame to include only MATLAB repositories
    matlab_df = df[df['language'] == 'MATLAB']
    
    # Preprocess the text data in the README files (use your preprocess_text_in_dataframe function)
    matlab_df = p.preprocess_text_in_dataframe(matlab_df, 'readme')
    
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Fit and transform the READMEs into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(matlab_df['readme'])
    
    # Get the feature names (words) corresponding to the columns of the TF-IDF matrix
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Calculate the mean TF-IDF score for each word across MATLAB READMEs
    mean_tfidf_scores = tfidf_matrix.mean(axis=0)
    
    # Convert the mean TF-IDF scores to a dictionary with words as keys and scores as values
    word_scores = {word: score for word, score in zip(feature_names, mean_tfidf_scores.tolist()[0])}
    
    # Find the top 3 most predictive words (highest TF-IDF scores)
    top_predictive_words = Counter(word_scores).most_common(3)
    
    # Display the top 3 most predictive words
    print("Top 3 Most Predictive Words in MATLAB READMEs:")
    for word, score in top_predictive_words:
        print(f"{word}: {score}")


def hypothesis_five(df):
    # Filter the DataFrame to include only TeX repositories
    tex_df = df[df['language'] == 'TeX']
    
    # Preprocess the text data in the README files (use your preprocess_text_in_dataframe function)
    tex_df = p.preprocess_text_in_dataframe(tex_df, 'readme')
    
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Fit and transform the READMEs into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(tex_df['readme'])
    
    # Get the feature names (words) corresponding to the columns of the TF-IDF matrix
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Calculate the mean TF-IDF score for each word across TeX READMEs
    mean_tfidf_scores = tfidf_matrix.mean(axis=0)
    
    # Convert the mean TF-IDF scores to a dictionary with words as keys and scores as values
    word_scores = {word: score for word, score in zip(feature_names, mean_tfidf_scores.tolist()[0])}
    
    # Find the top 3 most predictive words (highest TF-IDF scores)
    top_predictive_words = Counter(word_scores).most_common(3)
    
    # Display the top 3 most predictive words
    print("Top 3 Most Predictive Words in TeX READMEs:")
    for word, score in top_predictive_words:
        print(f"{word}: {score}")