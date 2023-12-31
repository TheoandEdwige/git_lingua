import numpy as np
import prepare as p
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer


def calculate_basic_statistics(dataframe, column_name):
    """
    Calculate basic statistics for the specified column.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column for which statistics are calculated.

    Returns:
        pandas.Series: Basic statistics (e.g., count, mean, std, min, 25%, 50%, 75%, max) for the specified column.
    """
    # Calculate basic statistics on the specified column
    basic_stats = dataframe[column_name].describe()
    return basic_stats

def identify_most_common_words(dataframe, column_name, top_n=10):
    """
    Identify the most common words in the specified column.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column with text data.
        top_n (int, optional): Number of most common words to identify (default is 10).

    Returns:
        pandas.Series: Top N most common words and their frequencies.
    """
    # Tokenize the text and count word frequencies
    all_text = " ".join(dataframe[column_name])
    words = all_text.split()
    word_counts = pd.Series(words).value_counts()
    # Get the top N most common words
    top_words = word_counts.head(top_n)
    return top_words

def analyze_readme_lengths(dataframe, column_name):
    """
    Analyze the distribution of README text lengths in the specified column.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column with text data.

    Returns:
        pandas.Series: A Series containing the lengths of README texts.
    """
    # Analyze the distribution of README lengths
    readme_lengths = dataframe[column_name].str.len()
    return readme_lengths

def generate_word_cloud(dataframe, column_name):
    """
    Generate a word cloud from the text in the specified column and display it.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column with text data.

    Returns:
        None
    """
    # Generate a word cloud from the specified column
    all_text = " ".join(dataframe[column_name])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def plot_readme_length_histogram(df):
    """
    Plot a histogram of README text lengths in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    readme_lengths = df['readme'].str.len()
    
    plt.figure(figsize=(10, 5))
    plt.hist(readme_lengths, bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('README Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of README Lengths')
    plt.show()

#visualization of the top words in readme
def top_words_barplot(top_words):
    """
    Create a bar plot to visualize the top words and their frequencies.

    Args:
        top_words (pandas.Series): Series containing top words and their frequencies.

    Returns:
        None
    """
    top_words.plot.barh()
    plt.show()


def idf_plot(n_documents):
    """
    plots the total numner and IDF for the most widely used words.
    
    n_documents: Total number of documents in your dataset.
    x: Number of documents a particular word appears in.  A single number.
    y: Inverse Document Frequency (IDF) of that word.
    """

    # n_documents = df.shape[0]

    x = np.arange(1, n_documents + 1)
    y = np.log(n_documents / x)

    plt.figure(figsize=(20, 10))
    plt.plot(x, y, marker='.')

    # plt.xticks(x)
    plt.xlabel('# of Documents the word appears in')
    plt.ylabel('IDF')
    plt.title('IDF for a given word')
    plt.show()



def hypothesis_one(df):
    """
    Visualize the impact of programming language on README word count.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
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
    """
    Visualize the relationship between README word count and the top 10 programming languages.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
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
    """
    Identify the top 3 most predictive words in R READMEs based on TF-IDF scores and visualize them.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
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

    # Create a bar plot to visualize the top 3 predictive words
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[word[0] for word in top_predictive_words], y=[word[1] for word in top_predictive_words])
    plt.title('Top 3 Most Predictive Words in R READMEs')
    plt.xlabel('Words')
    plt.ylabel('TF-IDF Score')
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()



def hypothesis_four(df):
    """
    Identify the top 3 most predictive words in MATLAB READMEs based on TF-IDF scores and visualize them.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
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

    # Create a bar chart to visualize the top 3 most predictive words
    words, scores = zip(*top_predictive_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, scores)
    plt.xlabel('Words')
    plt.ylabel('TF-IDF Scores')
    plt.title('Top 3 Most Predictive Words in MATLAB READMEs')
    plt.xticks(rotation=45)
    plt.show()



def hypothesis_five(df):
    """
    Identify the top 3 most predictive words in TeX READMEs based on TF-IDF scores and visualize them.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
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
    
    # Create a bar chart to visualize the top 3 most predictive words
    words, scores = zip(*top_predictive_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, scores)
    plt.xlabel('Words')
    plt.ylabel('TF-IDF Scores')
    plt.title('Top 3 Most Predictive Words in TeX READMEs')
    plt.xticks(rotation=45)
    plt.show()



def statistical_test1(filtered_df):
    """
    Perform a statistical test to compare word counts between Python and JavaScript READMEs.

    Args:
        filtered_df (pandas.DataFrame): The filtered DataFrame containing the data.

    Returns:
        None
    """
    # comparing Python and JavaScript word counts
    python_word_counts = filtered_df[filtered_df['language'] == 'Python']['readme_word_count']
    javascript_word_counts = filtered_df[filtered_df['language'] == 'JavaScript']['readme_word_count']
    
    # Perform Mann-Whitney U test
    statistic, p_value = mannwhitneyu(python_word_counts, javascript_word_counts)
    alpha=0.05
    # Check the result
    if p_value < alpha:
        print("Null hypothesis rejected: There is a significant difference in word counts.")
    else:
        print("We failed to reject the null hypothesis. There is no significant difference in word counts between programming languages in GitHub repositories.")
    # Display Chi2 and p
    print(f"Z-score: {statistic}")
    print(f"P-value: {p_value}")


def statistical_test2(df):
    """
    Perform a statistical test to analyze the association between programming language and specific word presence.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.

    Returns:
        None
    """
    # Create a DataFrame with counts of programming languages and word presence
    data = pd.crosstab(df['language'], df['UniqueWords'])
    
    # Perform the chi-squared test
    chi2, p, dof, expected = chi2_contingency(data)
    
    # Print the chi-squared statistic and p-value
    print(f"Chi-squared Statistic: {chi2}")
    print(f"P-Value: {p}")
    
    # Determine if the null hypothesis should be rejected
    alpha = 0.05  # Set your significance level
    if p < alpha:
        print("Reject the null hypothesis: There is an association between programming language and specific word presence.")
    else:
        print("Fail to reject the null hypothesis: No significant association found.")


def statistical_test3(df, word):
    """
    Perform a statistical test to analyze the association between the presence of a specific word and the choice of programming language.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        word (str): The specific word to analyze.

    Returns:
        None
    """
    # Create a binary column for the presence of 'word1' in READMEs
    df[word] = df['readme'].str.contains(word).astype(int)
    
    # Create a contingency table with presence/absence of 'word1' and programming languages
    contingency_table = pd.crosstab(df[word], df['language'])
    
    # Perform the chi-squared test
    chi2, p, _, _ = chi2_contingency(contingency_table)
    
    # Set the significance level (alpha)
    alpha = 0.05
    
    # Check if the p-value is less than alpha to determine significance
    if p < alpha:
        print(f"Reject the null hypothesis: The presence of '{word}' is associated with the choice of programming language.")
    else:
        print(f"Fail to reject the null hypothesis: The presence of '{word}' is not significantly associated with the choice of programming language.")
    # Display Chi2 and p
    print(f"Chi2: {chi2}")
    print(f"P-value: {p}")
