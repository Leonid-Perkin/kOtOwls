import sqlite3
def read_formulas_from_file(file_path):
    """Читает формулы и легенды из текстового файла"""
    formulas = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '//' in line:
                formula, legend = line.strip().split('//', 1)
                formulas.append((formula.strip(), legend.strip()))
    return formulas
def initialize_database():
    """Инициализация базы данных SQLite и добавление данных из текстового файла"""
    formulas_file = 'formulas.txt'
    formulas = read_formulas_from_file(formulas_file)
    connection = sqlite3.connect('formulas.db')
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS formulas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            formula TEXT NOT NULL,
            legend TEXT
        )
    """)
    cursor.executemany("""
        INSERT INTO formulas (formula, legend) VALUES (?, ?)
    """, formulas)
    
    connection.commit()
    cursor.close()
    connection.close()
initialize_database()