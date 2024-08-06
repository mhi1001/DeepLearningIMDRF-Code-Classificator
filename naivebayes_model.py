import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
import sys

# Load the LabelEncoder from BERT model
bert_model_dir = 'bert_model'
label_encoder = joblib.load(os.path.join(bert_model_dir, 'label_encoder.joblib'))

MODEL_DIR = 'naivebayes_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'naivebayes_model')

with open('message.json', 'r') as file:
    data = json.load(file)

if os.path.exists(MODEL_PATH):
    sys.exit('Model already exists')

data_df = pd.DataFrame(data)
data_df = data_df.explode('definition')
# Reset index
data_df = data_df.reset_index(drop=True)

data_df['code_encoded'] = label_encoder.fit_transform(data_df['code'])

X_train, X_test, y_train, y_test = train_test_split(data_df['definition'], data_df['code_encoded'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }

nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)
nb_scores = evaluate_model(y_test, nb_pred)

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(nb_model, os.path.join(MODEL_DIR, 'nb_model.joblib'))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))

print('Evaluation Metrics:')
for metric, score in nb_scores.items():
    print(f"{metric.capitalize()}: {score:.4f}")