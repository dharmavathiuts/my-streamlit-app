import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import streamlit as st

def run():
    st.title("Deep Learning Sentiment Analysis with LSTM")

    # Load dataset
    st.write("Loading dataset...")
    file_path = '/content/drive/My Drive/Reviews.csv'
    df = pd.read_csv(file_path)

    # Data Preprocessing
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        return text

    df['cleaned_text'] = df['Text'].apply(clean_text)

    # Tokenization
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(df['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])

    # Padding
    max_sequence_length = 100
    X = pad_sequences(sequences, maxlen=max_sequence_length)

    # Simplifying sentiment (Positive: 4-5, Negative: 1-3)
    df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
    y = df['sentiment'].values

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define LSTM Model
    model = Sequential()
    vocab_size = 5000
    embedding_dim = 128
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_shape=(max_sequence_length,)))
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Display model summary
    st.write("Model Summary:")
    model.summary(print_fn=lambda x: st.text(x))  # Print summary in Streamlit

    # Train the model
    st.write("Training the model...")
    history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    st.write(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot training and validation accuracy
    st.write("Accuracy over epochs:")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Train Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_title('Model Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='upper left')
    st.pyplot(fig)

    # Plot training and validation loss
    st.write("Loss over epochs:")
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Train Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Model Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper left')
    st.pyplot(fig)

    # Save the model (optional, not required in Streamlit)
    # model.save('lstm_sentiment_model.h5')

    # Prediction on new data
    st.write("Test the model with new reviews:")
    new_reviews = st.text_area("Enter reviews to analyze, separated by a newline.", "This product is amazing!\nI did not like this at all.")
    if new_reviews:
        new_reviews_list = new_reviews.split("\n")
        new_sequences = tokenizer.texts_to_sequences(new_reviews_list)
        new_padded = pad_sequences(new_sequences, maxlen=max_sequence_length)

        # Make predictions
        predictions = model.predict(new_padded)

        # Display predictions
        for review, pred in zip(new_reviews_list, predictions):
            sentiment = "Positive" if pred > 0.5 else "Negative"
            st.write(f"Review: {review} | Sentiment: {sentiment}")
