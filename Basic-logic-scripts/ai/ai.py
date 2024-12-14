import difflib
import sqlite3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("./formula_similarity_model")
model = AutoModelForSequenceClassification.from_pretrained("./formula_similarity_model")
def connect_to_db():
    """Подключение к базе данных SQLite"""
    connection = sqlite3.connect('formulas.db')
    return connection
def fetch_formulas(connection):
    """Извлекает формулы и их легенды из базы данных SQLite"""
    cursor = connection.cursor()
    cursor.execute("SELECT formula, legend FROM formulas")
    formulas = [(row[0], row[1]) for row in cursor.fetchall()]
    cursor.close()
    return formulas
def compare_formulas(formula1, formula2):
    """Сравнивает две формулы и возвращает коэффициент сходства с помощью difflib"""
    similarity = difflib.SequenceMatcher(None, formula1, formula2).ratio()
    return similarity
def neural_network_similarity(formula1, formula2):
    """Использует нейросеть для вычисления сходства между формулами"""
    inputs = tokenizer([formula1, formula2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        similarity_score = probabilities.max().item()
        
    return similarity_score
def check_formula_uniqueness(input_formula, connection):
    """Проверяет уникальность формулы"""
    formulas = fetch_formulas(connection)
    for formula, _ in formulas:
        if input_formula.strip() == formula.strip():
            print("Формула уже существует в базе данных. Оригинальность: 0.")
            return 0
    max_similarity_difflib = 0
    max_similarity_nn = 0
    most_similar_formula_difflib = None
    most_similar_formula_nn = None
    most_similar_legend_nn = None
    for formula, legend in formulas:
        similarity_difflib = compare_formulas(input_formula, formula)
        if similarity_difflib > max_similarity_difflib:
            max_similarity_difflib = similarity_difflib
            most_similar_formula_difflib = formula
        similarity_nn = neural_network_similarity(input_formula, formula)
        if similarity_nn > max_similarity_nn:
            max_similarity_nn = similarity_nn
            most_similar_formula_nn = formula
            most_similar_legend_nn = legend
    print("Результаты проверки уникальности:")
    print(f"Метод difflib - Максимальное сходство: {max_similarity_difflib * 100:.2f}%")
    print(f"Формула, наиболее похожая (difflib): {most_similar_formula_difflib}")
    print(f"Метод нейросети - Максимальное сходство: {max_similarity_nn * 100:.2f}%")
    return 0
input_formula = r"\frac{d}{dt} \left( \frac{1}{2} m*1 v^2 \right)"
try:
    connection = connect_to_db()
    check_formula_uniqueness(input_formula, connection)
except sqlite3.Error as err:
    print(f"Ошибка подключения к базе данных: {err}")
finally:
    if connection:
        connection.close()
