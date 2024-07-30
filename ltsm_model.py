import pandas as pd
import json
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


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
#labels = to_categorical(data_df['label'].values)
# Split data
#X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
