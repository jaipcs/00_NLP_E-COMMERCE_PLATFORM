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
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# ADVANCED CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    /* =======================================================
       MAIN BACKGROUND
    ======================================================= */

    .stApp {

        background:
        linear-gradient(
            135deg,
            #0F2027 0%,
            #203A43 50%,
            #2C5364 100%
        );

        color: white;

    }

    /* =======================================================
       REMOVE STREAMLIT DEFAULT PADDING
    ======================================================= */

    .block-container {

        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 4rem;
        padding-right: 4rem;

    }

    /* =======================================================
       MAIN TITLE
    ======================================================= */

    h1 {

        text-align: center;

        font-size: 60px !important;

        font-weight: 800 !important;

        color: #00FFD1 !important;

        margin-bottom: 5px;

        text-shadow:
            0px 0px 15px rgba(0,255,209,0.6);

    }

    /* =======================================================
       HEADERS
    ======================================================= */

    h2, h3 {

        color: #00FFD1 !important;

        font-weight: 700 !important;

    }

    /* =======================================================
       PARAGRAPH TEXT
    ======================================================= */

    p {

        color: #EAEAEA !important;

        font-size: 18px !important;

    }

    /* =======================================================
       SIDEBAR
    ======================================================= */

    section[data-testid="stSidebar"] {

        background:
        linear-gradient(
            180deg,
            #09121A 0%,
            #0F2027 100%
        );

        border-right:
            2px solid rgba(255,255,255,0.1);

    }

    /* Sidebar Text */

    section[data-testid="stSidebar"] * {

        color: white !important;

        font-size: 17px;

    }

    /* =======================================================
       RADIO BUTTONS
    ======================================================= */

    div[role="radiogroup"] label {

        background-color: rgba(255,255,255,0.05);

        padding: 12px;

        margin-bottom: 10px;

        border-radius: 12px;

        transition: 0.3s;

    }

    div[role="radiogroup"] label:hover {

        background-color: rgba(0,255,209,0.15);

    }

    /* =======================================================
       BUTTONS
    ======================================================= */

    .stButton>button {

        background:
        linear-gradient(
            90deg,
            #00FFD1,
            #00C9FF
        );

        color: black !important;

        border: none;

        border-radius: 15px;

        padding: 15px;

        font-size: 20px;

        font-weight: bold;

        width: 100%;

        transition: 0.3s;

        box-shadow:
            0px 0px 15px rgba(0,255,209,0.4);

    }

    .stButton>button:hover {

        transform: scale(1.03);

        box-shadow:
            0px 0px 25px rgba(0,255,209,0.7);

    }

    /* =======================================================
       TEXT AREA FIX
    ======================================================= */

    textarea {

        background-color: white !important;

        color: black !important;

        border-radius: 15px !important;

        border: 3px solid #00FFD1 !important;

        padding: 15px !important;

        font-size: 18px !important;

    }

    textarea::placeholder {

        color: gray !important;

    }

    /* =======================================================
       INPUT TEXT COLOR FIX
    ======================================================= */

    input {

        color: black !important;

    }

    /* =======================================================
       METRIC CARDS
    ======================================================= */

    div[data-testid="metric-container"] {

        background:
        rgba(255,255,255,0.08);

        border:
        1px solid rgba(255,255,255,0.1);

        padding: 20px;

        border-radius: 20px;

        box-shadow:
        0px 0px 15px rgba(0,0,0,0.3);

    }

    /* =======================================================
       DATAFRAME
    ======================================================= */

    .stDataFrame {

        border-radius: 15px;

        overflow: hidden;

    }

    /* =======================================================
       SUCCESS BOX
    ======================================================= */

    .stSuccess {

        border-radius: 15px !important;

    }

    /* =======================================================
       ERROR BOX
    ======================================================= */

    .stError {

        border-radius: 15px !important;

    }

    /* =======================================================
       INFO BOX
    ======================================================= */

    .stInfo {

        border-radius: 15px !important;

    }

    /* =======================================================
       CODE BLOCK
    ======================================================= */

    code {

        color: #00FFD1 !important;

        font-size: 15px !important;

    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE SECTION
# ============================================================

st.markdown(
    """
    <h1>
    🛒 E-Commerce Review Sentiment Analysis
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        text-align:center;
        font-size:24px;
        color:#EAEAEA;
        margin-bottom:40px;
    ">

    Predict customer review sentiment using
    NLP + Machine Learning 🚀

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
# LOAD MODEL
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
    "📌 Navigation",
    [
        "🏠 Home",
        "📊 Dataset",
        "🔮 Predict Sentiment"
    ]
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

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

        ### Technologies Used

        - NLP
        - Machine Learning
        - TF-IDF
        - Streamlit
        - Scikit-Learn
        """
    )

    st.success("✅ Model Loaded Successfully")


# ============================================================
# DATASET PAGE
# ============================================================

elif page == "📊 Dataset":

    st.header("📊 Dataset Overview")

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

elif page == "🔮 Predict Sentiment":

    st.header("🔮 Predict Review Sentiment")

    user_review = st.text_area(
        "Enter Customer Review",
        placeholder="Type customer review here..."
    )

    if st.button("🚀 Predict Sentiment"):

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

            st.markdown("<br>", unsafe_allow_html=True)

            if prediction == "positive":

                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(
                            90deg,
                            #00C853,
                            #64DD17
                        );

                        padding:30px;

                        border-radius:20px;

                        text-align:center;

                        font-size:35px;

                        font-weight:bold;

                        color:white;

                        box-shadow:
                        0px 0px 20px rgba(0,255,0,0.5);

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
                        background: linear-gradient(
                            90deg,
                            #D50000,
                            #FF1744
                        );

                        padding:30px;

                        border-radius:20px;

                        text-align:center;

                        font-size:35px;

                        font-weight:bold;

                        color:white;

                        box-shadow:
                        0px 0px 20px rgba(255,0,0,0.5);

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
                        background: linear-gradient(
                            90deg,
                            #2962FF,
                            #00B0FF
                        );

                        padding:30px;

                        border-radius:20px;

                        text-align:center;

                        font-size:35px;

                        font-weight:bold;

                        color:white;

                        box-shadow:
                        0px 0px 20px rgba(0,150,255,0.5);

                    ">

                    😐 NEUTRAL REVIEW

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("🧹 Cleaned Review")

            st.code(cleaned_review)
