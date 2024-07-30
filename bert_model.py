import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import json
import argparse
import os

######################Dont touch -  use gpu for faster training###################
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
print(f"Using device: {device}")
######################################################################################

with open('message.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 1 entry for 1 definition in  the dataset
#{'code': 'A0101', 'term': 'Patient-Device Incompatibility', 'definition': 'The patient experienced severe discomfort due to the device not aligning with their anatomy.'}
#{'code': 'A0101', 'term': 'Patient-Device Incompatibility', 'definition': "Complications arose because the device did not interact well with the patient's physiological condition."}
expanded_data = []
for entry in data:
    code = entry['code']
    term = entry['term']
    definitions = entry['definition']
    if not isinstance(definitions, list):
        definitions = [definitions]
    for definition in definitions:
        expanded_data.append({'code': code, 'term': term, 'definition': definition})

# Convert to DataFrame
data_df = pd.DataFrame(expanded_data)

# Combine term and definition into a single text field
data_df['text'] = data_df['term'] + ' ' + data_df['definition']

# Encode labels
label_to_id = {label: i for i, label in enumerate(data_df['code'].unique())}
data_df['label'] = data_df['code'].apply(lambda x: label_to_id[x])

# Tokenize text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data_df['tokens'] = data_df['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True))

# Split data
train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)

# BERT datatset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.labels = data['label'].values
        self.texts = data['tokens'].values
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {'input_ids': torch.tensor(self.texts[idx], dtype=torch.long), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}
        return item

train_dataset = Dataset(train_data)
test_dataset = Dataset(test_data)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load bert-base-uncased
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_to_id))

# Move model to GPU if available
model.to(device)

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training args
training_args = TrainingArguments(
    output_dir='./results',
    # Number of epochs
    num_train_epochs=10,  
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    learning_rate=1e-5,  
    weight_decay=0.01,
    logging_dir='./logs',
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

# Function to get top 5 predictions
def get_top_5_predictions(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)

    # Force model to use GPU...
    input_ids = torch.tensor([tokens]).to(device)  
    model.to(device)  
    model.eval()

    # Output top 5 codes for each query and their respective probability for debugging 
    with torch.no_grad():
        outputs = model(input_ids)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    top_5_probs, top_5_indices = torch.topk(probabilities, 5)
    top_5_codes = [list(label_to_id.keys())[i] for i in top_5_indices[0]]
    return top_5_codes, top_5_probs[0]

# Interaction loop
while True:
    text_input = input("Enter a description (or 'exit' to quit): ")
    if text_input.lower() == 'exit':
        break
    top_5_codes, top_5_probs = get_top_5_predictions(text_input)
    print("Top 5 codes:", top_5_codes)
    print("Probabilities:", top_5_probs)