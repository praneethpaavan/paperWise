### app.py (with Question Classification Integration)
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load classifier model and vectorizer
with open("models/classifier_model.pkl", "rb") as f:
    clf_model = pickle.load(f)
with open("models/vectorizer.pkl", "rb") as f:
    clf_vectorizer = pickle.load(f)

# Classification function
def classify_question(text):
    vec = clf_vectorizer.transform([text])
    prediction = clf_model.predict(vec)
    return prediction[0]

# Flask route to test classification
@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing 'question' field"}), 400

    category = classify_question(question)
    return jsonify({"question": question, "predicted_category": category})

# Sample test route
@app.route("/")
def home():
    return "PaperWise AI Classification API is running."

if __name__ == "__main__":
    app.run(debug=True)