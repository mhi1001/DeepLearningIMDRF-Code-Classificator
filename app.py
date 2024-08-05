from flask import Flask, render_template, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import json

app = Flask(__name__)

def get_texts(data):
    texts = []
    for entry in data:
        term = entry['term']
        definitions = entry['definition']
        if isinstance(definitions, list):
            for definition in definitions:
                texts.append(f"{term} {definition}")
        else:
            texts.append(f"{term} {definitions}")
    return texts


bert_model_path = 'bert_model/bert_model'
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

lstm_model_path = 'lstm_model/lstm_model.keras'
lstm_model = tf.keras.models.load_model(lstm_model_path)

with open('message.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

label_to_id = {entry['code']: i for i, entry in enumerate(data)}
id_to_label = {v: k for k, v in label_to_id.items()}

lstm_tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
texts = get_texts(data)
lstm_tokenizer.fit_on_texts(texts)

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
    else:
        return jsonify({'error': 'Invalid model type'})
    
    return jsonify({'codes': top_5_codes, 'probabilities': top_5_probs.tolist()})


def get_bert_predictions(text):
    tokens = bert_tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    input_ids = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        outputs = bert_model(input_ids)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    top_5_probs, top_5_indices = torch.topk(probabilities, 5)
    top_5_codes = [id_to_label[i.item()] for i in top_5_indices[0]]
    return top_5_codes, top_5_probs[0]

def get_lstm_predictions(text):
    sequence = lstm_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    pred_probs = lstm_model.predict(padded_sequence)[0]
    
    top_5_indices = pred_probs.argsort()[-5:][::-1]
    top_5_probs = pred_probs[top_5_indices]
    
    top_5_codes = [id_to_label[idx] for idx in top_5_indices]
    
    return top_5_codes, top_5_probs

if __name__ == '__main__':
    app.run(debug=True)