import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from textblob import TextBlob
import streamlit as st

def run():
    st.title("NLP Sentiment Analysis with LDA")

    # Load dataset
    df = pd.read_csv("C:/Users/shiva/Downloads/cleaned_reviews.csv")
    st.write("Data Preview:")
    st.dataframe(df.head())

    # Create 'Helpfulness' column
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'].replace(0, 1)

    # Display missing values
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Sentiment labeling
    def create_sentiment_label(score):
        if score >= 4:
            return 1  # Positive
        elif score == 3:
            return 0  # Neutral
        else:
            return -1  # Negative

    df['sentiment'] = df['Score'].apply(create_sentiment_label)
    st.write("Sentiment Distribution:")
    st.write(df['sentiment'].value_counts())

    # LDA topic modeling
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    text_vectorized = vectorizer.fit_transform(df['lemmatized_text'])

    lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
    lda_model.fit(text_vectorized)

    # Display top words in topics
    def display_topics(model, feature_names, no_top_words):
        topics = {}
        for topic_idx, topic in enumerate(model.components_):
            topics[f"Topic {topic_idx+1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        return topics

    no_top_words = 10
    lda_topics = display_topics(lda_model, vectorizer.get_feature_names_out(), no_top_words)
    st.write("LDA Topics with Top Words:")
    st.write(lda_topics)

    # Plot word clouds for topics
    st.write("Word Clouds for LDA Topics")
    for topic_idx, topic in enumerate(lda_model.components_):
        st.write(f"Topic {topic_idx + 1}")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
            ' '.join([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

    # Dominant topics
    doc_topic_dists = lda_model.transform(text_vectorized)
    df['dominant_topic'] = np.argmax(doc_topic_dists, axis=1)
    st.write("Dominant Topic Distribution:")
    st.write(df['dominant_topic'].value_counts())

    # Sentiment analysis using TextBlob
    df['textblob_sentiment'] = df['lemmatized_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    st.write("Sentiment Polarity Example:")
    st.dataframe(df[['lemmatized_text', 'textblob_sentiment']].head())

    # Plot sentiment distribution over topics
    topic_sentiment_scores = df.groupby('dominant_topic')['textblob_sentiment'].mean()
    st.write("Average Sentiment Scores by Dominant Topic:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=topic_sentiment_scores.index, y=topic_sentiment_scores.values, ax=ax)
    ax.set_title("Average Sentiment Scores by Topic")
    st.pyplot(fig)

    # Sentiment evolution over time
    df['Time'] = pd.to_datetime(df['Time'])
    df['Year'] = df['Time'].dt.year
    sentiment_over_time = df.groupby('Year')['textblob_sentiment'].mean()
    st.write("Sentiment Evolution Over Time:")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=sentiment_over_time.index, y=sentiment_over_time.values, marker='o', ax=ax)
    ax.set_title('Sentiment Evolution Over Time')
    st.pyplot(fig)
