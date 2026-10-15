from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

texts = [
    "this is amazing", "I love this", "that was awesome", "so happy right now",
    "this is the best", "you ate that", "that's fire", "no cap this is good",
    "it's giving iconic", "slay queen", "W friend", "big W", "this hits different",
    "I'm obsessed", "lowkey love this", "it's bussin", "that's a vibe",
    "this is terrible", "I hate this", "that was awful", "so disappointed",
    "this is the worst", "that's cringe", "L take", "big L", "that's mid",
    "this is giving flop", "I'm dead tired of this", "it's not it",
    "lowkey bad", "that's sus", "what a fail"
]

labels = ["positive"] * 17 + ["negative"] * (len(texts) - 17)

vectorizer = CountVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved!")
