import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import re
import html
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('spam.csv', encoding='latin1')

X = df.iloc[:, 1]
y = df.iloc[:, 0].map({'ham': 0, 'spam': 1})

def clean_text(text):
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X_cleaned = X.apply(clean_text)

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_cleaned)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

app = FastAPI()

class Item(BaseModel):
    prompt: str

@app.get('/')
def get():
    return {"message": "spam detection API is working"}

@app.post('/predict')
def get(req: Item):
    user_input = req.prompt
    new_user = clean_text(user_input)
    input_vector = vectorizer.transform([new_user])
    prediction = clf.predict(input_vector)[0]
    label = 'Spam' if prediction == 1 else 'Not Spam'
    return {"prediction": label}






