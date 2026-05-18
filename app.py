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

# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown(

    """
    <style>

    /* Main App Background */

    .stApp {

        background: linear-gradient(
            to right,
            #141E30,
            #243B55
        );

        color: white;

    }

    /* Main Title */

    h1 {

        color: #00FFD1;

        text-align: center;

        font-size: 55px;

        font-weight: bold;

        margin-bottom: 10px;

    }

    /* Sub Headers */

    h2, h3 {

        color: #00FFD1;

    }

    /* Sidebar */

    section[data-testid="stSidebar"] {

        background-color: #0B1320;

    }

    /* Sidebar Text */

    section[data-testid="stSidebar"] * {

        color: white;

    }

    /* Buttons */

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

    /* Button Hover */

    .stButton>button:hover {

        background-color: #00C9A7;

        color: white;

        transform: scale(1.02);

    }

    /* Text Area */

    textarea {

        border-radius: 12px !important;

        border: 2px solid #00FFD1 !important;

        background-color: rgba(255,255,255,0.05) !important;

        color: white !important;

    }

    /* Metric Cards */

    div[data-testid="metric-container"] {

        background-color: rgba(255,255,255,0.08);

        border: 1px solid rgba(255,255,255,0.2);

        padding: 18px;

        border-radius: 15px;

        backdrop-filter: blur(10px);

    }

    /* Dataframe */

    .stDataFrame {

        border-radius: 12px;

        overflow: hidden;

    }

    /* Success Message */

    .stSuccess {

        border-radius: 12px;

    }

    /* Error Message */

    .stError {

        border-radius: 12px;

    }

    /* Info Message */

    .stInfo {

        border-radius: 12px;

    }

    /* Code Block */

    code {

        color: #00FFD1 !important;

    }

    </style>
    """,

    unsafe_allow_html=True

)

)

# ============================================================
# CUSTOM TITLE
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
    <div style='
        text-align:center;
        font-size:22px;
        margin-bottom:35px;
        color:white;
    '>

    Predict customer review sentiment using
    NLP + Machine Learning 🚀

    </div>
    """,

    unsafe_allow_html=True

)




# ============================================================
# TITLE
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
    <div style='text-align:center;
                font-size:20px;
                margin-bottom:30px;'>

    Predict customer review sentiment using
    NLP and Machine Learning 🚀

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

            st.subheader("Cleaned Review")

            st.code(cleaned_review)
