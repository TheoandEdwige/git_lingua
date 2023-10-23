import pandas as pd
import matplotlib.pyplot as plt

from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer


def plot_most_common_words(df, num_words=20):
    # Combine all README texts into one string
    all_readme_text = ' '.join(df['readme'])
    
    # Tokenize the combined text
    words = all_readme_text.split()

    # Calculate word frequencies
    word_freq = FreqDist(words)

    # Plot the most common words
    word_freq.plot(num_words, title='Most Common Words in READMEs')
    plt.show()


def plot_unique_words_by_language(df):
    # Calculate the number of unique words in each README
    df['UniqueWords'] = df['readme'].apply(lambda x: len(set(x.split())))

    # Calculate the mean unique words by programming language
    unique_words_by_language = df.groupby('language')['UniqueWords'].mean()

    # Create a bar chart
    plt.figure(figsize=(10, 5))
    unique_words_by_language.plot(kind='bar', color='lightcoral')
    plt.title('Average Unique Words by Programming Language')
    plt.xlabel('Programming Language')
    plt.ylabel('Average Unique Words')
    plt.show()


def top_unique_words_by_language(df, num_languages=6, num_words=3):
    # Filter out rows where 'language' is not NaN
    df_filtered = df[df['language'].notna()]

    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

    # Fit and transform the READMEs into TF-IDF vectors
    tfidf_matrix = tfidf_vectorizer.fit_transform(df_filtered['readme'])

    # Create a DataFrame of the TF-IDF matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

    # Combine the TF-IDF DataFrame with the 'language' column
    tfidf_df['language'] = df_filtered['language'].values

    # Group by 'language' and calculate the mean TF-IDF score for each term
    unique_words_by_language = tfidf_df.groupby('language').mean()

    # Sort the terms by mean TF-IDF score
    unique_words_by_language = unique_words_by_language.T

    # Get the top N most popular programming languages based on the number of repositories
    top_languages = df_filtered['language'].value_counts().head(num_languages).index

    # Display the top M unique words for each of the top N most popular languages, excluding 'Jupyter Notebook'
    for language in top_languages:
        if language != 'Jupyter Notebook':
            top_words = unique_words_by_language[language].sort_values(ascending=False).head(num_words)
            print(f"Top {num_words} words for {language}:")
            print(top_words)
            print("\n")



def convert_and_dropna(df):
    #convert all column names to lowercase
    df.columns = [column.lower() for column in df.columns]
    
    # Remove rows with missing 'readme' values
    df = df.dropna(subset=['readme', 'language'])
    
    # Check for and remove duplicate rows
    df.drop_duplicates(subset=['name'], inplace=True)
    
    # Reset the index after dropping rows
    df = df.reset_index(drop=True)

    #convert all column names to lowercase
    df.columns = [column.lower() for column in df.columns]

    return df