import json
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

# Загрузка базы данных формул
def load_formulas(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Приведение формул к унифицированному виду
def preprocess_formula(formula):
    # Удаляем пробелы и стандартизируем переменные
    formula = re.sub(r"[a-zA-Z]", "x", formula.replace(" ", ""))
    return formula

# Подготовка данных для обучения
def prepare_data(formulas):
    processed = [preprocess_formula(f["latex"]) for f in formulas]
    unique_chars = sorted(set("".join(processed)))
    char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

    sequences = [[char_to_idx[ch] for ch in formula] for formula in processed]
    max_len = max(len(seq) for seq in sequences)
    sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

    return sequences_padded, char_to_idx, idx_to_char

# Создание нейросети
def create_model(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(128, return_sequences=True)),
        LSTM(128),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")  # Для оценки вероятности
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# Загрузка данных
formulas = load_formulas("formulas.json")
data, char_to_idx, idx_to_char = prepare_data(formulas)

# Разделение данных на обучение и тест
labels = np.zeros(len(data))  # Все формулы уникальны для начала
train_data, train_labels = data, labels

# Параметры модели
vocab_size = len(char_to_idx)
max_len = data.shape[1]

# Создание и обучение модели
model = create_model(vocab_size, max_len)
model.fit(train_data, train_labels, epochs=10, batch_size=16)

# Функция для проверки оригинальности
def check_originality(formula, model, char_to_idx, max_len):
    processed = preprocess_formula(formula)
    sequence = [char_to_idx.get(ch, 0) for ch in processed]  # 0 для неизвестных символов
    sequence_padded = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_len)
    prediction = model.predict(sequence_padded)[0][0]  # Вероятность в диапазоне [0, 1]
    originality_percentage = (1 - prediction) * 100  # Инверсия вероятности для оригинальности
    return f"Оригинальность формулы: {originality_percentage:.2f}%"

# Пример использования
new_formula = "E = mc^2"
print(check_originality(new_formula, model, char_to_idx, max_len))
