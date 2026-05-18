import streamlit as st
import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="E-Commerce Sentiment Analyzer",
    page_icon="🛒",
    layout="wide"

)

# ============================================================
# TITLE
# ============================================================

st.title("🛒 E-Commerce Review Sentiment Analysis")

st.write(

    """
    This app predicts customer review sentiment
    using NLP and Machine Learning.
    """

)

# ============================================================
# DOWNLOAD NLTK
# ============================================================

@st.cache_resource
def download_nltk():

    nltk.download("stopwords")

    nltk.download("punkt")

    nltk.download("punkt_tab")

download_nltk()

# ============================================================
# LOAD PICKLE FILES
# ============================================================

@st.cache_resource
def load_model():

    model = pickle.load(

        open("best_sentiment_model.pkl", "rb")

    )

    vectorizer = pickle.load(

        open("tfidf_vectorizer.pkl", "rb")

    )

    return model, vectorizer

model, tfidf = load_model()

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(

        "final_nlp_sentiment_eda_fe_dataset.csv"

    )

    return df

df = load_data()

# ============================================================
# TEXT CLEANING
# ============================================================

stop_words = set(stopwords.words("english"))

ps = PorterStemmer()

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z]", " ", text)

    words = word_tokenize(text)

    words = [

        word for word in words
        if word not in stop_words

    ]

    words = [

        ps.stem(word)
        for word in words

    ]

    return " ".join(words)

# ============================================================
# SIDEBAR
# ============================================================

page = st.sidebar.radio(

    "Navigation",

    [

        "Home",
        "Dataset",
        "Predict Sentiment"

    ]

)

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.header("📌 Project Overview")

    st.write(

        """
        This NLP application predicts customer review sentiment
        for e-commerce platforms like Amazon.

        The application uses:

        - TF-IDF Vectorization
        - Machine Learning Classification
        - NLP Text Cleaning
        - Sentiment Prediction

        """

    )

    st.success("Model Loaded Successfully")

# ============================================================
# DATASET PAGE
# ============================================================

elif page == "Dataset":

    st.header("📊 Dataset Overview")

    st.write("Dataset Shape:")

    st.write(df.shape)

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20))

    if "sentiment" in df.columns:

        st.subheader("Sentiment Distribution")

        st.bar_chart(

            df["sentiment"].value_counts()

        )

    if "rating" in df.columns:

        st.subheader("Rating Distribution")

        st.bar_chart(

            df["rating"].value_counts()

        )

# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "Predict Sentiment":

    st.header("🔮 Predict Review Sentiment")

    user_review = st.text_area(

        "Enter Customer Review",

        height=200

    )

    if st.button("Predict Sentiment"):

        if user_review.strip() == "":

            st.warning(

                "Please enter a review."

            )

        else:

            cleaned_review = clean_text(

                user_review

            )

            vector_input = tfidf.transform(

                [cleaned_review]

            )

            prediction = model.predict(

                vector_input

            )[0]

            st.subheader("Prediction Result")

            if prediction == "positive":

                st.success(

                    "😊 Positive Review"

                )

            elif prediction == "negative":

                st.error(

                    "😡 Negative Review"

                )

            else:

                st.info(

                    "😐 Neutral Review"

                )

            st.write(

                "Predicted Sentiment:",
                prediction

            )

            st.subheader("Cleaned Review")

            st.code(cleaned_review)
