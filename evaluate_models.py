import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load models and tokenizers
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

label_encoder = joblib.load('bert_model/label_encoder.joblib')

# Load test data
with open('newdata.json', 'r') as f:
    test_data = json.load(f)

def get_bert_prediction(text):
    tokens = bert_tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
    input_ids = torch.tensor([tokens]).to(device)
    
    with torch.no_grad():
        outputs = bert_model(input_ids)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    top_prediction = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
    
    return label_encoder.inverse_transform([top_prediction])[0]

def get_lstm_prediction(text):
    sequence = lstm_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    pred_probs = lstm_model.predict(padded_sequence)[0]
    
    top_prediction = np.argmax(pred_probs)
    
    return label_encoder.inverse_transform([top_prediction])[0]

def get_nb_prediction(text):
    text_vec = nb_vectorizer.transform([text])
    prediction = nb_model.predict(text_vec)[0]
    
    return label_encoder.inverse_transform([prediction])[0]

def evaluate_model(model_name, prediction_func):
    print(f"\n{model_name} Model Results:")
    print("CODE - Expected | Predicted | Result")
    print("-" * 40)
    
    y_true = []
    y_pred = []
    success_count = 0
    error_count = 0
    
    for item in test_data:
        text = item['definition']
        expected_code = item['code']
        predicted_code = prediction_func(text)
        
        result = "SUCCESS" if expected_code == predicted_code else "ERROR"
        if result == "SUCCESS":
            success_count += 1
        else:
            error_count += 1
        
        print(f"{expected_code:6} - {predicted_code:6} | {result}")
        
        y_true.append(expected_code)
        y_pred.append(predicted_code)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"\nTotal Predictions: {len(y_true)}")
    print(f"Successful Predictions: {success_count}")
    print(f"Errors: {error_count}")

# Evaluate each model
evaluate_model("BERT", get_bert_prediction)
evaluate_model("LSTM", get_lstm_prediction)
evaluate_model("Naive Bayes", get_nb_prediction)