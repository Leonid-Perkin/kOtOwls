import difflib

# Функция для чтения формул из текстового файла
def read_formulas(file_path):
    with open(file_path, 'r') as file:
        formulas = file.readlines()
    return [formula.strip() for formula in formulas if formula.strip()]

# Функция для вычисления сходства между двумя строками с использованием difflib
def compare_formulas(formula1, formula2):
    # Используем метод SequenceMatcher для оценки сходства строк
    similarity = difflib.SequenceMatcher(None, formula1, formula2).ratio()
    return similarity

# Основная функция для проверки оригинальности формул
def check_formula_uniqueness(input_formula, formulas_file):
    formulas = read_formulas(formulas_file)
    
    # Пройдемся по всем формулам и сравним с входной
    similarities = []
    for formula in formulas:
        similarity = compare_formulas(input_formula, formula)
        similarities.append((formula, similarity))
    
    # Выведем все результаты
    print("Сравнение формулы:")
    for formula, similarity in similarities:
        print(f"Формула: {formula}")
        print(f"Сходство: {similarity * 100:.2f}%")
        print("-" * 40)
    
    # Определим, является ли формула уникальной (например, если сходство < 80%)
    non_unique_formulas = [formula for formula, similarity in similarities if similarity > 0.8]
    if non_unique_formulas:
        print("Найдено схожие формулы. Возможно, они не уникальны.")
    else:
        print("Формула уникальна!")

# Пример использования
input_formula = r"\frac{d}{dt} \left( \frac{1}{2} m v^2 \right)"  # Входная формула
formulas_file = 'formulas.txt'  # Путь к файлу с формулами

check_formula_uniqueness(input_formula, formulas_file)
