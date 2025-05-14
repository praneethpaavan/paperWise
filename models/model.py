import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Check if training data exists
if not os.path.exists('training_data.csv'):
    raise FileNotFoundError("The dataset file 'training_data.csv' is missing.")

# Load training dataset
data = pd.read_csv('training_data.csv')

# Data Cleaning and Preprocessing
data['BloomLevel'] = data['BloomLevel'].str.upper()
data.dropna(subset=['Question', 'BloomLevel'], inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = ''.join([char.lower() for char in text if char not in string.punctuation])
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['Processed_Question'] = data['Question'].apply(preprocess_text)

# Encode Bloom levels
label_encoder = LabelEncoder()
data['Encoded_BloomLevel'] = label_encoder.fit_transform(data['BloomLevel'])

# Train-Test Split
X = data['Processed_Question']
y = data['Encoded_BloomLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a simple classifier (RandomForest)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_tfidf, y_train)

# Save models
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(clf, 'random_forest_classifier.pkl')

# Prediction and Evaluation
y_pred = clf.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
