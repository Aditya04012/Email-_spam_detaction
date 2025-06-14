import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import re
import html
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('spam.csv', encoding='latin1')[['v1', 'v2']]
df.columns = ['label', 'text']
df.dropna(inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X_cleaned = df['text'].apply(clean_text)
y = df['label']

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_cleaned)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train, y_train)

app = FastAPI()

class Item(BaseModel):
    prompt: str

@app.get('/')
def read_root():
    return {"message": "Spam detection API is working"}

@app.post('/predict')
def predict_spam(item: Item):
    cleaned = clean_text(item.prompt)
    vectorized = vectorizer.transform([cleaned])
    prediction = clf.predict(vectorized)[0]
    return {"prediction": "Spam" if prediction == 1 else "Not Spam"}






