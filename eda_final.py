import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load data and perform EDA when this function is called
def run():
    # Streamlit title for EDA section
    st.title("Exploratory Data Analysis")

    # Load dataset
    file_path = '/content/drive/My Drive/ass2_amazonfoodreview/Reviews.csv'
    df = pd.read_csv(file_path)

    # Show the first few rows
    st.write("First few rows of the dataset:")
    st.dataframe(df.head(10))

    # Show the description and summary info
    st.write("Dataset Description:")
    st.dataframe(df.describe())

    st.write("Dataset Info:")
    st.text(df.info())

    # Check for missing values
    st.write("Missing Values in Each Column:")
    st.write(df.isnull().sum())

    # Convert Time column to datetime format
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    st.write(f"Max Time: {df['Time'].max()}")

    # Word count in the reviews
    df['word_count'] = df['Text'].apply(lambda text: len(str(text).split()))
    st.write("Word Count Distribution:")
    st.dataframe(df['word_count'].describe())

    # Plotting Word Count Distribution
    st.write("Plot: Word Count Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['word_count'], bins=50, kde=True, ax=ax)
    ax.set_title("Word Count Distribution")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Boxplot of Score vs Word Count
    st.write("Boxplot: Score vs Word Count")
    fig, ax = plt.subplots()
    sns.boxplot(x="Score", y="word_count", width=0.5, data=df, ax=ax)
    ax.set_title('Boxplot Score vs Word Count')
    st.pyplot(fig)

    # Time distribution plot
    st.write("Plot: Review Time Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Time'], bins=50, kde=False, ax=ax)
    ax.set_title("Review Time Distribution")
    ax.set_xlabel("Time")
    ax.set_ylabel("No of Reviews")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Word Cloud of the most common words
    text = df['Text'].astype(str).str.lower().to_string()
    st.write("Word Cloud: Most Common Words")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Sentiment analysis using NLTK's VADER
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df['polarity'] = df['text_string'].apply(lambda x: analyzer.polarity_scores(x))
    df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x > 0 else 'neutral' if x == 0 else 'negative')

    # Sentiment countplot
    st.write("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(y='sentiment', data=df, ax=ax)
    st.pyplot(fig)

    # Correlation heatmap
    st.write("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, annot_kws={"fontsize":8}, ax=ax)
    st.pyplot(fig)
