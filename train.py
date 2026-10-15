from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

texts = [
    "I love this", "This is amazing", "I am happy", "Great product",
    "Very good", "I like it", "Excellent service", "Best experience",
    "I hate this", "This is terrible", "I am sad", "Bad product",
    "Very poor", "I dislike it", "Awful service", "Worst experience"
]

labels = [
    "positive", "positive", "positive", "positive",
    "positive", "positive", "positive", "positive",
    "negative", "negative", "negative", "negative",
    "negative", "negative", "negative", "negative"
]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(x, labels)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully!")
