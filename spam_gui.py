import pandas as pd
import re
import nltk
import tkinter as tk
from tkinter import messagebox

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean'] = df['message'].apply(clean_text)

# Convert to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean'])
y = df['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction function
def predict_message():
    msg = entry.get()

    if msg == "":
        messagebox.showwarning("Warning", "Enter a message!")
        return

    cleaned = clean_text(msg)
    vec = vectorizer.transform([cleaned])
    result = model.predict(vec)[0]

    if result == 1:
        output_label.config(text="🚨 Spam", fg="red")
    else:
        output_label.config(text="✅ Not Spam", fg="green")

# GUI
root = tk.Tk()
root.title("Spam Detector")
root.geometry("400x300")

tk.Label(root, text="Spam Detector", font=("Arial", 16)).pack(pady=10)

entry = tk.Entry(root, width=40)
entry.pack(pady=10)

tk.Button(root, text="Check", command=predict_message).pack(pady=10)

output_label = tk.Label(root, text="", font=("Arial", 14))
output_label.pack(pady=20)

root.mainloop()