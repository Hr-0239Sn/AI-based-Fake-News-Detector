from flask import Flask, render_template, request
import joblib, requests, re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

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

@app.route("/realtime")
def realtime():
    API_KEY = "8a33810ea3f74d308966d92d5e900c80"  # 🔴 Replace with your actual NewsAPI key
    url = f"https://newsapi.org/v2/everything?q=India&language=en&sortBy=publishedAt&apiKey={API_KEY}"

    response = requests.get(url)
    data = response.json()
    articles = data.get("articles", [])

    # Run prediction for each article using your model
    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        cleaned = clean_text(text)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        article["prediction"] = "REAL 📰✅" if pred == 1 else "FAKE 🚫🧢"

    return render_template("realtime.html", articles=articles)
    
if __name__ == "__main__":
    app.run(debug=True)
