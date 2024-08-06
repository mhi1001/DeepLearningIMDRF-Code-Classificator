from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import joblib
import numpy as np
import subprocess

app = Flask(__name__)


def check_model_exists(model_path):
    return os.path.exists(model_path)

def run_model_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(['python', script_name], capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
    else:
        print(f"{script_name} completed successfully.")
        print(result.stdout)

def ensure_models_exist():
    if not check_model_exists('bert_model/bert_model'):
        run_model_script('bert_model.py')
    
    if not check_model_exists('lstm_model/lstm_model.keras'):
        run_model_script('lstm_model.py')
    
    if not check_model_exists('naivebayes_model/nb_model.joblib'):
        run_model_script('naivebayes_model.py')


ensure_models_exist()

bert_model_path = 'bert_model/bert_model'
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

lstm_model_path = 'lstm_model/lstm_model.keras'
lstm_model = tf.keras.models.load_model(lstm_model_path)
lstm_tokenizer = joblib.load('lstm_model/lstm_tokenizer.joblib')

nb_model = joblib.load('naivebayes_model/nb_model.joblib')
nb_vectorizer = joblib.load('naivebayes_model/tfidf_vectorizer.joblib')

# Load LabelEncoder
label_encoder = joblib.load('bert_model/label_encoder.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    model_type = request.json['model_type']
    
    if model_type == 'bert':
        top_5_codes, top_5_probs = get_bert_predictions(text)
    elif model_type == 'lstm':
        top_5_codes, top_5_probs = get_lstm_predictions(text)
    elif model_type == 'naivebayes':
        top_5_codes, top_5_probs = get_nb_predictions(text)
    else:
        return jsonify({'error': 'Invalid model type'})
    
    top_5_codes = top_5_codes.tolist() if hasattr(top_5_codes, 'tolist') else list(top_5_codes)
    top_5_probs = top_5_probs.tolist() if hasattr(top_5_probs, 'tolist') else list(top_5_probs)
    
    return jsonify({'codes': top_5_codes, 'probabilities': top_5_probs})


def get_bert_predictions(text):
    tokens = bert_tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    input_ids = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        outputs = bert_model(input_ids)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    top_5_probs, top_5_indices = torch.topk(probabilities, 5)
    
    # Move to CPU cuz LE does not like GPU
    top_5_indices = top_5_indices.cpu().numpy()[0]
    top_5_probs = top_5_probs.cpu().numpy()[0]
    
    top_5_codes = label_encoder.inverse_transform(top_5_indices)
    return top_5_codes, top_5_probs

def get_lstm_predictions(text):
    sequence = lstm_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    pred_probs = lstm_model.predict(padded_sequence)[0]
    
    top_5_indices = pred_probs.argsort()[-5:][::-1]
    top_5_probs = pred_probs[top_5_indices]
    
    top_5_codes = label_encoder.inverse_transform(top_5_indices)
    
    return top_5_codes, top_5_probs

def get_nb_predictions(text):
    text_vec = nb_vectorizer.transform([text])
    probabilities = nb_model.predict_proba(text_vec)[0]
    
    top_5_indices = np.argsort(probabilities)[::-1][:5]
    top_5_codes = label_encoder.inverse_transform(top_5_indices)
    top_5_probs = probabilities[top_5_indices]
    
    return top_5_codes, top_5_probs

if __name__ == '__main__':
    app.run(debug=True)