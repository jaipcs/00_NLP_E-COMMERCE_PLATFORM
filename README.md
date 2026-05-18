# 🛒 E-Commerce Review Sentiment Analysis

An NLP-based Sentiment Analysis Web Application built using:

- Python
- Natural Language Processing (NLP)
- Machine Learning
- TF-IDF Vectorization
- Streamlit

This application predicts whether customer reviews are:

- 😊 Positive
- 😐 Neutral
- 😡 Negative

---

# 📌 Project Overview

This project analyzes customer reviews from e-commerce platforms such as Amazon and predicts sentiment using Machine Learning and NLP techniques.

The workflow includes:

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. TF-IDF Vectorization
5. Model Training & Evaluation
6. Streamlit Deployment

---

# 🚀 Features

## ✅ NLP Preprocessing

- Lowercasing
- Stopword Removal
- Tokenization
- Stemming
- Regex Cleaning

---

## ✅ EDA Features

- Sentiment Distribution
- Rating Distribution
- Word Count Analysis
- Word Clouds
- Top Frequent Words
- Bigram & Trigram Analysis
- Emoji Analysis
- Review Length Analysis

---

## ✅ Machine Learning Models

The project experimented with multiple NLP classification models:

- Multinomial Naive Bayes
- Complement Naive Bayes
- Bernoulli Naive Bayes
- Logistic Regression
- SGD Classifier
- Passive Aggressive Classifier
- Ridge Classifier
- Linear SVC
- Random Forest
- Extra Trees Classifier

---

## ✅ Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

---

# 📂 Project Structure

```text
nlp-ecommerce/
│
├── app.py
├── requirements.txt
├── README.md
├── best_sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── final_nlp_sentiment_eda_fe_dataset.csv
```

---

# 🧠 Machine Learning Pipeline

```text
Raw Review
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Machine Learning Model
   ↓
Sentiment Prediction
```

---

# 📊 Dataset

The dataset contains:

- Review Title
- Review Body
- Rating
- Sentiment Labels

Sentiment labels are generated using ratings:

| Rating | Sentiment |
|---|---|
| 4-5 | Positive |
| 3 | Neutral |
| 1-2 | Negative |

---

# 🔮 Prediction Example

### Input Review

```text
The product quality is amazing and delivery was very fast.
```

### Predicted Output

```text
Positive 😊
```

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/nlp-ecommerce.git
```

Move into the project folder:

```bash
cd nlp-ecommerce
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

# 🌐 Streamlit Deployment

This project is deployed using:

:contentReference[oaicite:0]{index=0}

Deployment steps:

1. Push all files to GitHub
2. Open Streamlit Cloud
3. Connect GitHub repository
4. Select `app.py`
5. Deploy

---

# 📈 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Streamlit | Web Application |
| Scikit-learn | Machine Learning |
| NLTK | NLP Preprocessing |
| Pandas | Data Analysis |
| Matplotlib | Visualization |
| Seaborn | Visualization |

---

# 🧪 NLP Techniques Used

- Tokenization
- Stopword Removal
- Stemming
- TF-IDF
- N-Grams
- Text Cleaning

---

# 📌 Future Improvements

- Deep Learning Models
- BERT Transformers
- Real-time Review Scraping
- Fake Review Detection
- Aspect-Based Sentiment Analysis
- Multilingual Sentiment Analysis

---

# 👨‍💻 Author

Developed as an NLP & Machine Learning project for E-Commerce Sentiment Analysis.

---

# ⭐ If you like this project

Give this repository a ⭐ on GitHub.
