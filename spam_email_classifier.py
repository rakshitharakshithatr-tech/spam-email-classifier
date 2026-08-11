import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

data = pd.read_csv(
    url,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Convert labels into numbers
data["label_num"] = data["label"].map({"ham": 0, "spam": 1})

# Separate messages and labels
X = data["message"]
y = data["label_num"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Test model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test new messages
new_messages = [
    "Congratulations! You have won a free lottery prize. Click now!",
    "Hi, please send me the notes for tomorrow's class.",
    "URGENT! You have won $1000. Claim your prize now!",
    "Can we meet at 5 pm today?"
]

new_messages_tfidf = vectorizer.transform(new_messages)
predictions = model.predict(new_messages_tfidf)

print("\nNew Message Predictions:")

for message, prediction in zip(new_messages, predictions):
    if prediction == 1:
        print("🚨 SPAM:", message)
    else:
        print("✅ NOT SPAM:", message)
