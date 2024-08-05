import pandas as pd
import json
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

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

MODEL_DIR = 'lstm_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras')


data_df = pd.DataFrame(expanded_data)

# Combine term and definition into a single text field
data_df['text'] = data_df['term'] + ' ' + data_df['definition']

# Encode labels
label_to_id = {label: i for i, label in enumerate(data_df['code'].unique())}
data_df['label'] = data_df['code'].apply(lambda x: label_to_id[x])

print(f"[DEBUG] {data_df['label']}")
# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data_df['text'].values)

# Padding needed to have same length LTSM quirks
sequences = tokenizer.texts_to_sequences(data_df['text'].values)
padded_sequences = pad_sequences(sequences, maxlen=100)

print(f"[DEBUG] {padded_sequences}")
# Prepare labels
labels = to_categorical(data_df['label'].values)
# Split data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

if os.path.exists(MODEL_PATH):
    print("Loading existing LSTM model...")
    model = tensorflow.keras.models.load_model(MODEL_PATH)
    print(f"[DEBUG] {model.summary()}")
else:
    print("Training new LSTM model...")

    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    model.add(Dense(len(label_to_id), activation='softmax'))

    print(f"[DEBUG] {model.summary()}")

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy}')

    # Accuracy, losss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Get top 5 codes - with percentage for debugging
def get_top_5_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    pred_probs = model.predict(padded_sequence)[0]
    
    top_5_indices = pred_probs.argsort()[-5:][::-1]
    top_5_probs = pred_probs[top_5_indices]
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    top_5_codes = [id_to_label[idx] for idx in top_5_indices]
    
    return top_5_codes, top_5_probs

# Interaction loop
while True:
    text_input = input("Enter a description (or 'exit' to quit): ")
    if text_input.lower() == 'exit':
        break
    top_5_codes, top_5_probs = get_top_5_predictions(text_input)
    print("Top 5 codes:", top_5_codes)
    print("Probabilities:", top_5_probs)