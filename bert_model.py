import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import json
import os
import joblib

######################Dont touch -  use gpu for faster training###################
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
print(f"Using device: {device}")
######################################################################################

MODEL_DIR = "bert_model"
MODEL_PATH = os.path.join(MODEL_DIR, "bert_model")

# Load dataset
with open("dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

data_df = pd.DataFrame(data)
print(data_df)
data_df = data_df.explode("definition")
# Reset index
data_df = data_df.reset_index(drop=True)
print(data_df)

# visualize dataset info
print("Dataset Overview:")
print(f"Total samples: {len(data_df)}")
print(f"Unique codes: {data_df['code'].nunique()}")

data_df["text_length"] = data_df["definition"].str.len()
data_df["word_count"] = data_df["definition"].str.split().str.len()

# plt.figure(figsize=(12, 6))
# data_df['code'].value_counts().plot(kind='bar')
# plt.title('Distribution of Codes')
# plt.xlabel('Code')
# plt.ylabel('Count')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig('code_distribution.png')
# plt.close()

# plt.figure(figsize=(10, 6))
# sns.histplot(data_df['text_length'], kde=True)
# plt.title('Distribution of Text Lengths')
# plt.xlabel('Text Length (Number of Characters in Each Definition)')
# plt.ylabel('Frequency (Number of Definitions with This Length)')
# plt.tight_layout()
# plt.savefig('text_length_distribution.png')
# plt.close()

# plt.figure(figsize=(12, 6))
# sns.boxplot(x='code', y='word_count', data=data_df)
# plt.title('Word Count Distribution by Code')
# plt.xlabel('Code')
# plt.ylabel('Word Count')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig('word_count_distribution.png')
# plt.close()

# Encode the "code" labels
label_encoder = LabelEncoder()
data_df["label"] = label_encoder.fit_transform(data_df["code"])

# Tokenize text
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
data_df["tokens"] = data_df["definition"].apply(
    lambda x: tokenizer.encode(
        x, add_special_tokens=True, max_length=512, truncation=True
    )
)

# Split data
train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)


# BERT datatset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.labels = data["label"].values
        self.texts = data["tokens"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.texts[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


train_dataset = Dataset(train_data)
test_dataset = Dataset(test_data)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load bert-base-uncased
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_encoder.classes_)
)

# Move model to GPU if available
model.to(device)


# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


if os.path.exists(MODEL_PATH):
    print("Loading existing BERT model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
else:
    print("Training new BERT model...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=len(label_encoder.classes_)
    )
    model.to(device)
    # Training args
    training_args = TrainingArguments(
        output_dir="./results",
        # Number of epochs
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        report_to="none",
        # Evaluate at each epoch
        evaluation_strategy="epoch",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate the model
    trainer.train()
    results = trainer.evaluate()
    print("Evaluation results:", results)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_PATH)
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))
    print(f"Encoders saved to {MODEL_PATH}")
