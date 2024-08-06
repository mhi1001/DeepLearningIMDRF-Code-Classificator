import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load models and tokenizers
bert_model_path = "bert_model/bert_model"
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

lstm_model_path = "lstm_model/lstm_model.keras"
lstm_model = tf.keras.models.load_model(lstm_model_path)
lstm_tokenizer = joblib.load("lstm_model/lstm_tokenizer.joblib")

nb_model = joblib.load("naivebayes_model/nb_model.joblib")
nb_vectorizer = joblib.load("naivebayes_model/tfidf_vectorizer.joblib")

label_encoder = joblib.load("bert_model/label_encoder.joblib")

# Load test data
with open("newdata.json", "r") as f:
    test_data = json.load(f)


def get_next_evaluation_number():
    i = 1
    while os.path.exists(f"evaluation_{i}"):
        i += 1
    return i


def create_evaluation_folder(eval_number):
    folder_name = f"evaluation_{eval_number}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def get_bert_prediction(text):
    tokens = bert_tokenizer.encode(
        text, add_special_tokens=True, max_length=512, truncation=True
    )
    input_ids = torch.tensor([tokens]).to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    top_prediction = torch.argmax(probabilities, dim=1).cpu().numpy()[0]

    return label_encoder.inverse_transform([top_prediction])[0]


def get_lstm_prediction(text):
    sequence = lstm_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    # verbose 0 remove the annoying green bar for each predict
    pred_probs = lstm_model.predict(padded_sequence, verbose=0)[0]

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
        text = item["definition"]
        expected_code = item["code"]
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    print("\nMetrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"\nTotal Predictions: {len(y_true)}")
    print(f"Successful Predictions: {success_count}")
    print(f"Errors: {error_count}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "success_count": success_count,
        "error_count": error_count,
    }


def plot_metrics(bert_results, lstm_results, nb_results, folder):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

    data = [
        [bert_results[m] for m in ["accuracy", "precision", "recall", "f1"]],
        [lstm_results[m] for m in ["accuracy", "precision", "recall", "f1"]],
        [nb_results[m] for m in ["accuracy", "precision", "recall", "f1"]],
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, data[0], width, label="BERT")
    ax.bar(x, data[1], width, label="LSTM")
    ax.bar(x + width, data[2], width, label="Naive Bayes")

    ax.set_ylabel("Scores")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(folder, "metrics_comparison.png"))
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, folder):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(folder, f"confusion_matrix_{model_name}.png"))
    plt.close()


def plot_error_analysis(bert_results, lstm_results, nb_results, folder):
    models = ["BERT", "LSTM", "Naive Bayes"]
    correct = [
        bert_results["success_count"],
        lstm_results["success_count"],
        nb_results["success_count"],
    ]
    incorrect = [
        bert_results["error_count"],
        lstm_results["error_count"],
        nb_results["error_count"],
    ]
    total = [sum(pair) for pair in zip(correct, incorrect)]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, correct, width, label="Correct", color="g")
    rects2 = ax.bar(x + width / 2, incorrect, width, label="Incorrect", color="r")

    ax.set_ylabel("Number of Predictions")
    ax.set_title("Error Analysis")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    def autolabel(rects, values):
        for rect, value in zip(rects, values):
            height = rect.get_height()
            ax.annotate(
                f"{value}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1, correct)
    autolabel(rects2, incorrect)

    plt.tight_layout()
    plt.savefig(os.path.join(folder, "error_analysis.png"))
    plt.close()


eval_number = get_next_evaluation_number()
eval_folder = create_evaluation_folder(eval_number)

bert_results = evaluate_model("BERT", get_bert_prediction)
lstm_results = evaluate_model("LSTM", get_lstm_prediction)
nb_results = evaluate_model("Naive Bayes", get_nb_prediction)

plot_metrics(bert_results, lstm_results, nb_results, eval_folder)
plot_confusion_matrix(
    bert_results["y_true"], bert_results["y_pred"], "BERT", eval_folder
)
plot_confusion_matrix(
    lstm_results["y_true"], lstm_results["y_pred"], "LSTM", eval_folder
)
plot_confusion_matrix(
    nb_results["y_true"], nb_results["y_pred"], "Naive Bayes", eval_folder
)
plot_error_analysis(bert_results, lstm_results, nb_results, eval_folder)
