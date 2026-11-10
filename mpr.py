import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset directly from a public URL
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_table(url, header=None, names=['label', 'message'])

# 2. Split features and labels
X = df['message']
y = df['label']  # 'ham' (legitimate) or 'spam'

# 3. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Extraction (TF-IDF with stopword removal)
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# 5. Model Training (Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Evaluation
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Test sample messages
samples = [
    "Congratulations! You won a $1,000 Walmart gift card. Click here to claim.",
    "Hey, are we still meeting in the library at 4 PM today?"
]
preds = model.predict(tfidf.transform(samples))
for text, pred in zip(samples, preds):
    print(f"\nMessage: '{text}' --> Prediction: {pred.upper()}")