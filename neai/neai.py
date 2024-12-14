import difflib
import sqlite3

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
    """Сравнивает две формулы и возвращает коэффициент сходства"""
    similarity = difflib.SequenceMatcher(None, formula1, formula2).ratio()
    return similarity

def check_formula_uniqueness(input_formula, connection):
    """Проверяет уникальность формулы"""
    formulas = fetch_formulas(connection)
    for formula, _ in formulas:
        if input_formula.strip() == formula.strip():
            print("Формула уже существует в базе данных. Оригинальность: 0.")
            return 0
    max_similarity = 0
    most_similar_formula = None
    most_similar_legend = None
    for formula, legend in formulas:
        similarity = compare_formulas(input_formula, formula)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_formula = formula
            most_similar_legend = legend
    print(f"Максимальное сходство: {max_similarity * 100:.2f}%")
    print(f"Максимально совпадающая формула: {most_similar_formula}")
    print(f"Легенда для наиболее схожей формулы: {most_similar_legend}")
    print(f"Легенда: Формула, наиболее схожая с введенной, имеет коэффициент сходства {max_similarity * 100:.2f}% и является наиболее вероятным кандидатом на повторение.")
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