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
# CUSTOM CSS STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #141E30, #243B55);
        color: white;
    }

    h1 {
        color: #00FFD1;
        text-align: center;
        font-size: 55px;
        font-weight: bold;
        margin-bottom: 10px;
    }

    h2, h3 {
        color: #00FFD1;
    }

    section[data-testid="stSidebar"] {
        background-color: #0B1320;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .stButton>button {
        background-color: #00FFD1;
        color: black;
        border-radius: 12px;
        height: 3.2em;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #00C9A7;
        color: white;
        transform: scale(1.02);
    }

    textarea {
        border-radius: 12px !important;
        border: 2px solid #00FFD1 !important;
        background-color: rgba(255,255,255,0.05) !important;
        color: white !important;
    }

    div[data-testid="metric-container"] {
        background-color: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.2);
        padding: 18px;
        border-radius: 15px;
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    code {
        color: #00FFD1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <h1>🛒 E-Commerce Review Sentiment Analysis</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align:center; font-size:22px; margin-bottom:35px; color:white;'>
        Predict customer review sentiment using NLP + Machine Learning 🚀
    </div>
    """,
    unsafe_allow_html=True
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
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model():
    with open("best_sentiment_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


model, tfidf = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("final_nlp_sentiment_eda_fe_dataset.csv")
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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Reviews", df.shape[0])

    with col2:
        st.metric("Total Columns", df.shape[1])

    with col3:
        st.metric("Model Status", "Loaded")

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

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

    if "sentiment" in df.columns:
        st.subheader("Sentiment Distribution")
        st.bar_chart(df["sentiment"].value_counts())

    if "rating" in df.columns:
        st.subheader("Rating Distribution")
        st.bar_chart(df["rating"].value_counts())


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
            st.warning("Please enter a customer review.")

        else:
            cleaned_review = clean_text(user_review)

            vector_input = tfidf.transform([cleaned_review])

            prediction = model.predict(vector_input)[0]

            st.markdown(
                """
                <h2 style='text-align:center;'>Prediction Result</h2>
                """,
                unsafe_allow_html=True
            )

            if prediction == "positive":
                st.markdown(
                    """
                    <div style="
                        background-color:#00C853;
                        padding:25px;
                        border-radius:15px;
                        text-align:center;
                        font-size:30px;
                        font-weight:bold;
                        color:white;
                    ">
                        😊 POSITIVE REVIEW
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            elif prediction == "negative":
                st.markdown(
                    """
                    <div style="
                        background-color:#D50000;
                        padding:25px;
                        border-radius:15px;
                        text-align:center;
                        font-size:30px;
                        font-weight:bold;
                        color:white;
                    ">
                        😡 NEGATIVE REVIEW
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    """
                    <div style="
                        background-color:#2962FF;
                        padding:25px;
                        border-radius:15px;
                        text-align:center;
                        font-size:30px;
                        font-weight:bold;
                        color:white;
                    ">
                        😐 NEUTRAL REVIEW
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.subheader("Predicted Class")
            st.write(prediction)

            st.subheader("Cleaned Review")
            st.code(cleaned_review)
