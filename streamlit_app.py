import streamlit as st
import classification_models  # Import Classification Models script
import eda_final  # Import Exploratory Data Analysis script
import nlpsentimentanalysislda  # Import NLP Sentiment Analysis script
import deep_learning  # Import Deep Learning script

# Streamlit app title
st.title("Data Science and Machine Learning Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", 
                        ["Classification Models", "Exploratory Data Analysis", "NLP Sentiment Analysis", "Deep Learning"])

# Display the selected page
if page == "Classification Models":
    st.header("Classification Models")
    classification_models.run()  # Call the run function from classification_models.py

elif page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    eda_final.run()  # Call the run function from eda_final.py

elif page == "NLP Sentiment Analysis":
    st.header("NLP Sentiment Analysis with LDA")
    nlpsentimentanalysislda.run()  # Call the run function from nlpsentimentanalysislda.py

elif page == "Deep Learning":
    st.header("Deep Learning")
    deep_learning.run()  # Call the run function from deep_learning.py
