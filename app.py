import nltk
nltk.download('stopwords')
from flask import Flask, render_template, request
import joblib
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text Cleaner
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

import re

def clean_text(text):
    text = text.lower()  
    text = re.sub(r"http\S+", "", text)  
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    tokens = text.split()  
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words] 
    return " ".join(tokens)  

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    clean_news = clean_text(news)
    vect = vectorizer.transform([clean_news])
    prediction = model.predict(vect)
    result = "REAL News 📰✅" if prediction[0] == 1 else "FAKE News 🚫🧢"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
