import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download("stopwords")


fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")


fake_df["label"] = 0
real_df["label"] = 1


data = pd.concat([fake_df, real_df])
data = data.sample(frac=1).reset_index(drop=True)


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


data["text"] = data["text"].apply(clean_text)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=10000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = PassiveAggressiveClassifier()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n📊 Accuracy: {round(acc * 100, 2)}%")
print("\n🧾 Confusion Matrix:\n", cm)

def predict_news(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return "REAL News 📰✅" if prediction[0] == 1 else "FAKE News 🚫🧢"


sample = input("\n🧾 Enter News Text to Check: ")
result = predict_news(sample)
print("\n📢 Prediction:", result)

import joblib


joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
import joblib


