import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

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


data_df = pd.DataFrame(expanded_data)

# Combine term and definition into a single text field
data_df['text'] = data_df['term'] + ' ' + data_df['definition']

# Encode labels
label_to_id = {label: i for i, label in enumerate(data_df['code'].unique())}
data_df['label'] = data_df['code'].apply(lambda x: label_to_id[x])

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data_df['text'].values)

# Padding needed to have same length LTSM quirks
sequences = tokenizer.texts_to_sequences(data_df['text'].values)
padded_sequences = pad_sequences(sequences, maxlen=100)

# Prepare labels
labels = to_categorical(data_df['label'].values)
# Split data
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(len(label_to_id), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))


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