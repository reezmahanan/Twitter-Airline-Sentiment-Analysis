# ✈️ Twitter US Airline Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle%20Airline%20Sentiment-lightgrey.svg)](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)

An end-to-end Natural Language Processing (NLP) pipeline to analyze and classify passenger sentiment on Twitter toward major US airlines (Virgin America, United, Delta, Southwest, American, and US Airways).

---

## 📌 Project Overview

Customer feedback on social media is crucial for airline operations and customer support. This project tackles the challenge of automatically classifying tweets into three sentiment categories:
- **Negative** (~63%)
- **Neutral** (~21%)
- **Positive** (~16%)

### Key Highlights of the Architecture
- **Leak-Free Pipeline**: Train/Test splitting is performed **before** vectorization so test data never influences the TF-IDF vocabulary or IDF weights.
- **Sentiment-Aware Text Cleaning**: Custom cleaning pipeline that eliminates noise (`@mentions`, URLs, punctuation) while **preserving critical negation words** (`not`, `no`, `never`, `can't`, `didn't`, etc.), preventing phrases like *"not happy"* from becoming *"happy"*.
- **Class-Imbalance Handling**: Uses stratified splitting and balanced class weighting (`class_weight='balanced'`).
- **Multi-Model Benchmark**: Evaluates and compares four distinct classifiers across Accuracy, Macro F1, and Weighted F1 metrics.
- **Interactive Predictor**: Ready-to-use function to predict sentiment and output class confidence probabilities for custom tweets.

---

## 📊 Dataset

The dataset is sourced from the [Crowdflower Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) collection on Kaggle:
- **Total Records**: 14,640 tweets
- **Features Used**: `text` (tweet content) and `airline_sentiment` (target label: `negative`, `neutral`, `positive`)
- **Metadata Explored**: `airline` (airline name), `negativereason` (reason for negative sentiment), `negativereason_confidence`

---

## 🔬 Pipeline & Workflow

```
Raw Tweets (14,640)
       │
       ▼
1. Exploratory Data Analysis (EDA)
   ├── Sentiment distribution (~63% Negative class imbalance)
   ├── Airline-by-airline sentiment comparison
   └── Top negative reasons (Late Flights, Customer Service, Cancelled Flights)
       │
       ▼
2. Sentiment-Aware Preprocessing
   ├── Lowercasing & URL removal
   ├── @mention removal (avoids airline handle bias)
   ├── Negation word preservation
   └── Porter Stemming & Tokenization
       │
       ▼
3. Stratified Train-Test Split (80/20)
   └── Prevents data leakage and maintains class proportions
       │
       ▼
4. Bi-gram TF-IDF Vectorization
   └── max_features=5000, ngram_range=(1, 2)
       │
       ▼
5. Model Benchmarking
   ├── Multinomial Naive Bayes
   ├── Logistic Regression (balanced)
   ├── Linear Support Vector Classifier (LinearSVC)
   └── Random Forest Classifier
       │
       ▼
6. Evaluation & Confusion Matrix Heatmaps
       │
       ▼
7. Serialization & Custom Tweet Inference
   └── Exported with joblib (`best_sentiment_model.pkl`, `tfidf_vectorizer.pkl`)
```

---

## 📈 Model Performance Benchmark

Models were evaluated using **Macro F1-score** alongside Accuracy to fairly evaluate performance across the imbalanced classes:

| Model | Accuracy | Macro F1 | Weighted F1 | Strengths |
| :--- | :---: | :---: | :---: | :--- |
| **Linear SVC** | **~79.5%** | **~0.74** | **~0.79** | Best margin separation on sparse TF-IDF |
| **Logistic Regression (Balanced)** | **~78.2%** | **~0.73** | **~0.78** | Calibrated probabilities, robust across all classes |
| **Random Forest** | ~75.5% | ~0.67 | ~0.74 | Strong non-linear baseline |
| **Multinomial Naive Bayes** | ~72.1% | ~0.61 | ~0.70 | Fast baseline, tends to favor majority class |

> **Note:** Because negative tweets represent ~63% of the dataset, Macro F1 is the most reliable metric for judging neutral and positive detection capability.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Install Required Packages
```bash
pip install kagglehub nltk scikit-learn seaborn matplotlib pandas numpy joblib
```

### 3. Run the Jupyter Notebook
Open the notebook in your preferred environment (Jupyter Lab, VS Code, or Google Colab):
```bash
jupyter notebook Twitter_Airline_Sentiment_Analysis.ipynb
```

---

## 💡 Custom Tweet Prediction Example

You can test custom tweets directly using the interactive function included in the notebook:

```python
from Twitter_Airline_Sentiment_Analysis import predict_airline_sentiment

# Example 1: Positive Tweet
predict_airline_sentiment("@VirginAmerica your staff was incredible and the flight was on time! Best airline ever!")
# Output: Prediction: [POSITIVE] | Confidence: {'negative': '4.2%', 'neutral': '8.1%', 'positive': '87.7%'}

# Example 2: Negative Tweet with Negation
predict_airline_sentiment("@AmericanAir not happy with the cancelled flight and lost baggage.")
# Output: Prediction: [NEGATIVE] | Confidence: {'negative': '93.5%', 'neutral': '4.1%', 'positive': '2.4%'}

# Example 3: Neutral Inquiry
predict_airline_sentiment("@SouthwestAir what time does flight 432 arrive at JFK?")
# Output: Prediction: [NEUTRAL] | Confidence: {'negative': '12.0%', 'neutral': '82.3%', 'positive': '5.7%'}
```

---

## 📁 Repository Structure

```
Twitter sentiment analysis/
│
├── Twitter_Airline_Sentiment_Analysis.ipynb   # Complete analysis & modeling notebook
├── README.md                                  # Project documentation
├── best_sentiment_model.pkl                   # Exported trained model (generated on run)
└── tfidf_vectorizer.pkl                       # Exported TF-IDF vectorizer (generated on run)
```

---

## 🔮 Future Improvements

- [ ] **Transformer Models**: Fine-tune `cardiffnlp/twitter-roberta-base-sentiment` or `distilbert-base-uncased` for 88%+ accuracy.
- [ ] **Aspect-Based Sentiment Analysis (ABSA)**: Predict specific failure points (luggage, delays, seating) alongside sentiment.
- [ ] **Interactive Web UI**: Build a Streamlit or Gradio dashboard for real-time sentiment scoring.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
