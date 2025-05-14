import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample data — replace with your dataset
data = pd.DataFrame({
    'question': [
        "Define Ohm's Law.",
        "Explain the process of photosynthesis.",
        "What is a binary search tree?",
        "Write a note on World War II."
    ],
    'label': ['Physics', 'Biology', 'Computer Science', 'History']
})

# Preprocessing
X = data['question']
y = data['label']

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train classifier
model = LogisticRegression()
model.fit(X_vec, y)

# Save the model and vectorizer
with open("classifier_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved!")
