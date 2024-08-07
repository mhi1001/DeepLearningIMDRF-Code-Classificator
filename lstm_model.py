import pandas as pd
import json
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import joblib
import os, sys, io

# subprocess related shenanigans - probably bc of the progress bar for each epoch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

with open("dataset.json", "r", encoding="utf-8") as file:
    data = json.load(file)

MODEL_DIR = "lstm_model"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")


data_df = pd.DataFrame(data)
data_df = data_df.explode("definition")
# Reset index
data_df = data_df.reset_index(drop=True)

# Load encodings from bert
label_encoder = joblib.load("encoder\label_encoder.joblib")

# Encode labels using the loaded LabelEncoder
data_df["label"] = label_encoder.transform(data_df["code"])
print(f"[DEBUG] {data_df['label']}")

# Tokenize text
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(data_df["definition"].values)

# Padding needed to have same length lstm quirks
sequences = tokenizer.texts_to_sequences(data_df["definition"].values)
padded_sequences = pad_sequences(sequences, maxlen=200)

print(f"[DEBUG] {padded_sequences}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, data_df["label"].values, test_size=0.2, random_state=42
)

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
    model.add(Dense(len(label_encoder.classes_), activation="softmax"))

    print(f"[DEBUG] {model.summary()}")

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
    # sparse_categorical_crossentropy -> no onehotencondig needed - label is sufficient
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=14,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    tokenizer_path = os.path.join(MODEL_DIR, "lstm_tokenizer.joblib")
    joblib.dump(tokenizer, tokenizer_path)
    print(f"Model saved to {MODEL_PATH}")
