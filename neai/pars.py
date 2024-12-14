import requests
from bs4 import BeautifulSoup
import sqlite3

# Создание базы данных и таблицы
DB_NAME = "formulas.db"
def initialize_database():
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS formulas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            formula TEXT NOT NULL,
            legend TEXT
        )
    """)
    connection.commit()
    connection.close()

# Функция для сохранения формул в базу данных
def save_to_database(formulas):
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()
    cursor.executemany("INSERT INTO formulas (formula, legend) VALUES (?, ?)", formulas)
    connection.commit()
    connection.close()

# Парсинг сайта и извлечение формул
def parse_formulas(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Ошибка загрузки страницы: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    formulas = []

    # Структура сайта: <h3 style="color:#408020;">Formula Title</h3>, <img src="...">, <textarea name="tex">...</textarea>
    formula_title = soup.find("h3", style="color:#408020;").text if soup.find("h3", style="color:#408020;") else None
    latex_code = soup.find("textarea", {"name": "tex"}).text if soup.find("textarea", {"name": "tex"}) else None
    
    if formula_title and latex_code:
        formulas.append((latex_code.strip(), formula_title.strip()))

    return formulas

# Основная часть
if __name__ == "__main__":
    initialize_database()

    # Итерация по страницам с формулами
    all_formulas = []
    for i in range(1, 101):  # Перебор страниц с номерами от 1 до 100
        url = f"https://equationsheet.com/eqninfo/Equation-{i:04d}.html"
        print(f"Обрабатывается страница: {url}")
        formulas = parse_formulas(url)
        all_formulas.extend(formulas)

    print(f"Найдено формул: {len(all_formulas)}")
    save_to_database(all_formulas)
    print("Формулы успешно сохранены в базу данных.")
