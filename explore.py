import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

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

<<<<<<< HEAD
#visualization of the top words in readme
def top_words_barplot(top_words):
    top_words.plot.barh()gi
    plt.show()
=======
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

    

>>>>>>> 0b639d397bf522f5618a700c6ca8ab5f54dc2999
