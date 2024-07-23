import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import matplotlib.pyplot as plt
import json
import numpy as np
from scipy import stats

# Check CUDA availability and set device
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Load and preprocess data
with open('message.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

expanded_data = []
for entry in data:
    code = entry['code']
    term = entry['term']
    definitions = entry['definition']
    if not isinstance(definitions, list):
        definitions = [definitions]
    for definition in definitions:
        expanded_data.append({'code': code, 'term': term, 'definition': definition})

data_df = pd.DataFrame(expanded_data)
data_df['text'] = data_df['term'] + ' ' + data_df['definition']

label_to_id = {label: i for i, label in enumerate(data_df['code'].unique())}
data_df['label'] = data_df['code'].apply(lambda x: label_to_id[x])

# Tokenize text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 512
data_df['tokens'] = data_df['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=max_length, truncation=True))

# Dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.labels = data['label'].values
        self.texts = data['tokens'].values
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {'input_ids': torch.tensor(self.texts[idx], dtype=torch.long), 'labels': torch.tensor(self.labels[idx], dtype=torch.long)}
        return item

# Compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    return {
        'accuracy': acc,
        'f1': f1,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall
    }

# Train and evaluate function
def train_and_evaluate(model_name, train_dataset, eval_dataset, num_labels):
    if 'roberta' in model_name:
        model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=f'./results_{model_name}',
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,  # if your GPU supports it
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    return trainer.evaluate()

# Cross-validation function
def cross_validate(model_name, data, num_labels, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = []
    for train_index, val_index in kf.split(data):
        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]
        train_dataset = Dataset(train_data)
        val_dataset = Dataset(val_data)
        cv_results.append(train_and_evaluate(model_name, train_dataset, val_dataset, num_labels))
    return cv_results

# Visualization function
def plot_results(results):
    metrics = list(results[list(results.keys())[0]][0].keys())
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.boxplot([
            [run[metric] for run in results[model]]
            for model in results
        ], labels=list(results.keys()))
        plt.title(f'Comparison of {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Statistical comparison function
def compare_models_statistically(cv_results, metric='accuracy'):
    model_scores = {model: [result[metric] for result in results] for model, results in cv_results.items()}
    for model1 in model_scores:
        for model2 in model_scores:
            if model1 != model2:
                t_stat, p_value = stats.ttest_ind(model_scores[model1], model_scores[model2])
                print(f"{model1} vs {model2} ({metric}):")
                print(f"t-statistic: {t_stat}, p-value: {p_value}")
                print()

# Main execution
if __name__ == "__main__":
    # Models to compare
    models_to_compare = ['bert-base-uncased', 'bert-large-uncased', 'roberta-base']
    
    # Perform cross-validation for each model
    cv_results = {model: cross_validate(model, data_df, len(label_to_id)) for model in models_to_compare}
    
    # Plot results
    plot_results(cv_results)
    
    # Statistical comparison
    compare_models_statistically(cv_results)
    
    # Train the best model on the full dataset
    best_model = max(cv_results, key=lambda x: np.mean([run['accuracy'] for run in cv_results[x]]))
    print(f"Best model: {best_model}")
    
    full_dataset = Dataset(data_df)
    final_results = train_and_evaluate(best_model, full_dataset, full_dataset, len(label_to_id))
    print("Final model performance:", final_results)
    
    # Save the best model
    model = BertForSequenceClassification.from_pretrained(best_model, num_labels=len(label_to_id))
    model.save_pretrained('./best_model')
    tokenizer.save_pretrained('./best_model')
    
    # Function to get top 5 predictions
    def get_top_5_predictions(text):
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)
        input_ids = torch.tensor([tokens]).to(device)
        model.to(device)
        model.eval()
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