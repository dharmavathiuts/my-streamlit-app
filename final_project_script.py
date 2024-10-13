
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re

# Load Dataset
df = pd.read_csv(r"C:\Users\Aditi Vyas\downloads\Reviews.csv")

# EDA: Exploratory Data Analysis
def eda(df):
    # Basic statistics
    print(df.describe())
    
    # Missing values
    print(df.isnull().sum())
    
    # Visualizations
    sns.histplot(df['Score'], bins=5)
    plt.title('Distribution of Review Scores')
    plt.show()
    
    sns.histplot(df['HelpfulnessNumerator'], bins=50)
    plt.title('Distribution of Helpfulness Numerator')
    plt.show()

# NLP: Topic Modeling and Word Clouds
def nlp_analysis(df):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    # Preprocess text
    df['cleaned_text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))
    
    # Vectorize text using CountVectorizer
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    text_vectorized = vectorizer.fit_transform(df['cleaned_text'])
    
    # Topic Modeling (LDA)
    lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
    lda_model.fit(text_vectorized)
    
    # Display word clouds for each topic
    def display_word_cloud(lda_model, feature_names, num_words=10):
        for topic_idx, topic in enumerate(lda_model.components_):
            wordcloud = WordCloud(background_color='white').generate(' '.join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Topic {topic_idx + 1}')
            plt.show()

    display_word_cloud(lda_model, vectorizer.get_feature_names_out())

# Classification Models
def classification_models(df):
    # Preprocess text for models
    df['cleaned_text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))
    
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df['cleaned_text'])
    
    # Use Score as labels (for simplicity)
    y = df['Score'].apply(lambda x: 1 if x >= 4 else 0)  # Binary classification: positive (1), negative (0)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}")
    
    # Decision Tree Classifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)}")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")

# Deep Learning: LSTM Model for Sentiment Analysis
def deep_learning_model(df):
    df['cleaned_text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))
    
    # Tokenization and Padding
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    max_length = 100
    X = pad_sequences(sequences, maxlen=max_length)
    
    # Sentiment as labels
    y = df['Score'].apply(lambda x: 1 if x > 3 else 0)  # Binary sentiment: positive (1) or negative (0)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define LSTM model
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_length))
    model.add(LSTM(units=128))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"LSTM Test Accuracy: {accuracy}")
    
    # Visualize training history
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.show()

# Main Function
if __name__ == "__main__":
    # Load and analyze the data
    eda(df)
    nlp_analysis(df)
    classification_models(df)
    deep_learning_model(df)
