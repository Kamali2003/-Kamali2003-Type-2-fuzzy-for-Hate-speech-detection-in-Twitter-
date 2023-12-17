!pip install fuzzywuzzy scikit-learn

import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Function to apply fuzzy logic to classify tweets
def apply_fuzzy_logic(tweet, keywords):
    for keyword, threshold in keywords.items():
        if fuzz.ratio(tweet, keyword) >= threshold:
            return keyword
    return "Other"

data = pd.read_csv('/content/drive/MyDrive/Dataset/Twitter_Data.csv')

# Define fuzzy logic keywords and thresholds
fuzzy_keywords = {
    'positive': 70,
    'negative': 70,
    'neutral': 50
}

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('stopwords')
nltk.download('punkt')

import numpy as np

# Remove missing values (NaN)
data = data[data['clean_text'].notna()]

# Convert non-string values to strings
data['clean_text'] = data['clean_text'].apply(lambda x: str(x))

import pandas as pd

# Remove rows with missing values from the entire DataFrame
data.dropna(inplace=True)

# To remove missing values from a specific column, for example 'label'
data.dropna(subset=['clean_text'], inplace=True)

text_data = data['clean_text']
labels = data['category']

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # Join the tokens back into a string
    return ' '.join(tokens)

# Apply the preprocessing function to the text data
text_data = text_data.apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=10)

# Fit and transform the preprocessed text data to create the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

data['sentiment'] = data['clean_text'].apply(lambda x: apply_fuzzy_logic(x, fuzzy_keywords))

X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

def classify_tweet(tweet):
    # You may need to adjust the threshold based on your data
    hate_threshold = 70
    non_hate_threshold = 40

    # Example criteria for fuzzy matching
    hate_keywords = ["hate", "racist", "offensive"]
    non_hate_keywords = ["happy", "neutral", "positive"]

    # Check for hate
    for keyword in hate_keywords:
        if fuzz.ratio(tweet, keyword) >= hate_threshold:
            return 'hate'

    # Check for non-hate
    for keyword in non_hate_keywords:
        if fuzz.ratio(tweet, keyword) >= non_hate_threshold:
            return 'non-hate'

    # If neither hate nor non-hate, classify as neutral
    return 'neutral'

data['predicted_label'] = data['clean_text'].apply(classify_tweet)

X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['predicted_label'], test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_vectorized, y_train)

y_pred = rf.predict(X_test_vectorized)

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:\n', classification_rep)
