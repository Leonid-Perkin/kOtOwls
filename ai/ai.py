import sympy as sp
import json
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from difflib import SequenceMatcher
import re

class FormulaDataset(Dataset):
    """
    Класс для обработки данных формул для обучения нейросети.
    """
    def __init__(self, formulas, tokenizer, max_length=128):
        self.formulas = formulas
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        formula1, formula2, label = self.formulas[idx]
        inputs = self.tokenizer(
            formula1, formula2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_formulas(filepath):
    """
    Загружает формулы из JSON файла.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            formulas = json.load(file)
            if not formulas:
                print("Формулы не найдены в базе.")
            return formulas
    except FileNotFoundError:
        print("Файл с формулами не найден.")
        return []
    except json.JSONDecodeError:
        print("Ошибка чтения базы формул.")
        return []

def save_formulas(filepath, formulas):
    """
    Сохраняет формулы в JSON файл.
    """
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(formulas, file, ensure_ascii=False, indent=4)

def clean_formula(formula_str):
    """
    Очистка строки формулы от символов, которые могут вызвать ошибки при парсинге.
    """
    formula_str = formula_str.replace('^', '**')  # Преобразуем степень в Python-формат
    formula_str = formula_str.replace('infinity', 'sp.oo')  # Используем правильный символ для бесконечности
    return formula_str

def normalize_formula(formula_str):
    """
    Преобразует строку формулы в математическое выражение SymPy.
    """
    try:
        # Очищаем строку формулы перед нормализацией
        cleaned_formula = clean_formula(formula_str)
        
        # Определяем символы (переменные) для формулы
        variables = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        symbols = sp.symbols(variables)
        
        # Преобразуем очищенную формулу в выражение SymPy
        expr = sp.sympify(cleaned_formula, locals={var: symbol for var, symbol in zip(variables, symbols)})
        return expr
    except Exception as e:
        print(f"Ошибка нормализации формулы: {e}")
        return None

def calculate_similarity(formula1, formula2):
    """
    Рассчитывает схожесть между двумя формулами на основе математического анализа.
    """
    try:
        if formula1.equals(formula2):  # Используем equals для сравнения объектов SymPy
            return 1.0  # Полное совпадение
        return 0.0
    except Exception as e:
        print(f"Ошибка сравнения формул: {e}")
        return 0.0

def calculate_neural_similarity(input_formula, database, model, tokenizer):
    """
    Рассчитывает схожесть формулы на основе нейросетевой модели.
    """
    try:
        inputs = tokenizer([input_formula] + [entry['formula'] for entry in database], padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        scores = outputs.logits.softmax(dim=-1)[:, 1].tolist()  # Предполагаем, что индекс 1 - схожесть
        return max(scores[1:], default=0.0)
    except Exception as e:
        print(f"Ошибка нейросетевого анализа: {e}")
        return 0.0

def calculate_originality(input_formula, database, model=None, tokenizer=None):
    """
    Рассчитывает оригинальность введенной формулы с использованием символьного и нейросетевого анализа.
    """
    input_expr = normalize_formula(input_formula)
    if input_expr is None:
        return 100  # Если формулу нельзя обработать, считаем ее уникальной

    # Нормализуем формулы из базы
    normalized_database = [
        normalize_formula(entry['formula']) for entry in database
    ]

    # Символьный анализ
    similarity_scores = [
        calculate_similarity(input_expr, db_expr)
        for db_expr in normalized_database if db_expr is not None
    ]

    max_symbolic_similarity = max(similarity_scores, default=0)

    # Нейросетевой анализ (если модель и токенизатор переданы)
    if model and tokenizer:
        neural_similarity = calculate_neural_similarity(input_formula, database, model, tokenizer)
    else:
        neural_similarity = 0.0

    # Итоговая оригинальность
    max_similarity = max(max_symbolic_similarity, neural_similarity)
    return 100 - max_similarity * 100

def train_neural_model(train_data, tokenizer, model_name="distilbert-base-uncased", epochs=3):
    """
    Обучение нейросетевой модели на парах формул.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    dataset = FormulaDataset(train_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Эпоха {epoch + 1}/{epochs}, Средняя потеря: {total_loss / len(dataloader):.4f}")

    return model

def generate_training_data(database):
    """
    Генерирует пары формул для обучения модели.
    """
    training_data = []
    for i, entry1 in enumerate(database):
        for j, entry2 in enumerate(database):
            if i != j:
                label = 1 if normalize_formula(entry1['formula']) == normalize_formula(entry2['formula']) else 0
                training_data.append((entry1['formula'], entry2['formula'], label))
    return training_data

def main():
    """
    Основная функция программы.
    """
    database_path = "formulas.json"

    # Загружаем базу формул
    database = load_formulas(database_path)

    if not database:
        print("База формул пуста или отсутствует.")
        return

    # Загрузка токенизатора
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Генерация данных для обучения
    train_data = generate_training_data(database)
    print(f"Сгенерировано {len(train_data)} пар данных для обучения.")

    # Обучение модели
    model = train_neural_model(train_data, tokenizer, model_name)

    print("Добро пожаловать в систему проверки оригинальности формул!")
    print("Введите формулу:")

    while True:
        input_formula = input("Формула: ")
        if input_formula.lower() in ["exit", "выход"]:
            print("Выход из программы.")
            break

        originality = calculate_originality(input_formula, database, model, tokenizer)
        print(f"Оригинальность формулы: {originality:.2f}%")

        # Добавить новую формулу в базу, если она уникальна
        if originality > 0:
            add_to_db = input("Добавить формулу в базу? (да/нет): ").lower()
            if add_to_db in ["да", "yes"]:
                # Проверка на уникальность перед добавлением
                if all(normalize_formula(input_formula) != normalize_formula(entry['formula']) for entry in database):
                    database.append({"id": len(database) + 1, "formula": input_formula})
                    save_formulas(database_path, database)
                    print("Формула добавлена в базу.")
                else:
                    print("Формула уже существует в базе.")

if __name__ == "__main__":
    main()
