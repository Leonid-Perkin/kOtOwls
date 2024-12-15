# views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import *

import os
from PIL import Image
from pix2tex.cli import LatexOCR
import warnings
import logging

from sympy import symbols, latex, parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
import re

import difflib
import sqlite3
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
def connect_to_db():
    """Подключение к базе данных SQLite"""
    connection = sqlite3.connect('main/formulas.db')
    return connection

def save_formula_to_db(latex_text):
    """Сохранение формулы в базу данных с полем legend='no'"""
    connection = connect_to_db()
    cursor = connection.cursor()

    # Проверка наличия таблицы перед её созданием
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS formulas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            formula TEXT NOT NULL,
            legend TEXT
        )
    ''')

    # SQL запрос для вставки данных
    cursor.execute('''
        INSERT INTO formulas (formula, legend)
        VALUES (?, ?)
    ''', (latex_text, 'no'))

    connection.commit()
    connection.close()
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
        embeddings = outputs.last_hidden_state.mean(dim=1)
        similarity_score = cosine_similarity(embeddings.cpu().numpy())[0][1]
    return similarity_score
def check_formula_uniqueness(input_formula, connection):
    """Проверяет уникальность формулы"""
    formulas = fetch_formulas(connection)
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
            most_similar_legend_difflib = legend
        similarity_nn = neural_network_similarity(input_formula, formula)
        if similarity_nn > max_similarity_nn:
            max_similarity_nn = similarity_nn
            most_similar_formula_nn = formula
            most_similar_legend_nn = legend
    return {
        'max_similarity_difflib': max_similarity_difflib * 100,
        'most_similar_formula_difflib': most_similar_formula_difflib,
        'most_similar_legend_difflib': most_similar_legend_difflib,
        'max_similarity_nn': max_similarity_nn * 100,
        'most_similar_formula_nn': most_similar_formula_nn,
        'most_similar_legend_nn': most_similar_legend_nn
    }

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("pydantic").setLevel(logging.CRITICAL)

def extract_latex_from_image(image_path):
    try:
        model = LatexOCR()
        image = Image.open(image_path).convert('RGB')
        latex_formula = model(image)
        if not latex_formula.strip().startswith("\\begin{array}"):
            latex_formula = f"\\begin{{equation}}\n{latex_formula}\n\\end{{equation}}"
        return latex_formula
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None
    
def preprocess_expression(expression):
    replacements = {
        r'\btg\b': 'tan',  # Замена tg -> tan
        r'\bctg\b': 'cot', # Замена ctg -> cot
        r'\bsh\b': 'sinh', # Замена sh -> sinh
        r'\bch\b': 'cosh', # Замена ch -> cosh
        r'\bth\b': 'tanh', # Замена th -> tanh
    }

    for pattern, replacement in replacements.items():
        expression = re.sub(pattern, replacement, expression)
    expression = expression.replace('^', '**').strip()
    return expression

def text_to_latex(input_text):
    """
    Преобразует текстовые формулы в формат LaTeX.

    :param input_text: Обычный текст с формулами
    :return: Текст с формулами в формате LaTeX
    """
    try:
        transformations = [
            (r'\b([a-zA-Z])\^2\b', r'\1^2'),  # x^2 -> x^2
            (r'\b([a-zA-Z])_([a-zA-Z0-9]+)\b', r'\1_\2'),  # x_1 -> x_1
            (r'\bsqrt\(([a-zA-Z0-9+\-*/^ ]+)\)', r'\\sqrt{\1}'),  # sqrt(x) -> \sqrt{x}
            (r'\b([a-zA-Z]+)\(([a-zA-Z0-9+\-*/^ ]+)\)', r'\1(\2)'),  # sin(x) -> sin(x)
            (r'\b([0-9]+)\*([a-zA-Z])\b', r'\1 \\cdot \2'),  # 2*x -> 2 \cdot x
            (r'\b([a-zA-Z0-9]+)\s*\*\s*([a-zA-Z0-9]+)\b', r'\1 \\cdot \2'),  # x * y -> x \cdot y
            (r'([0-9]+)\^([0-9]+)', r'\1^{\2}'),  # 2^3 -> 2^{3}
            (r'\b([0-9]+)\s*/\s*([0-9]+)\b', r'\\frac{\1}{\2}'),  # 2/3 -> \frac{2}{3}
            (r'\b([a-zA-Z]+)\s*/\s*([a-zA-Z]+)\b', r'\\frac{\1}{\2}'),  # x/y -> \frac{x}{y}
            (r'\b([a-zA-Z0-9]+)\s*\+\s*([a-zA-Z0-9]+)\b', r'\1 + \2'),  # x + y -> x + y
            (r'\b([a-zA-Z0-9]+)\s*-\s*([a-zA-Z0-9]+)\b', r'\1 - \2'),  # x - y -> x - y
            (r'\b([a-zA-Z])\s*=\s*([a-zA-Z0-9+\-*/^ ]+)\b', r'\1 = \2'),  # x = y -> x = y
            (r'\bintegral\s*\(([a-zA-Z0-9+\-*/^ ]+)\)\s*d([a-zA-Z])', r'\\int {\1} \, d\2'),  # integral(f(x)) dx -> \int {f(x)} \, dx
            (r'\bsum\s*\(([a-zA-Z0-9+\-*/^ ]+),([a-zA-Z0-9]+)=([a-zA-Z0-9]+)\.\.([a-zA-Z0-9]+)\)', r'\\sum_{\2=\3}^{\4} {\1}'),  # sum(f(x), i=1..n) -> \sum_{i=1}^{n} {f(x)}
            (r'\blim\s*\(([a-zA-Z0-9+\-*/^ ]+),\s*([a-zA-Z0-9]+)->([a-zA-Z0-9]+)\)', r'\\lim_{\2 \to \3} {\1}'),  # lim(f(x), x->a) -> \lim_{x \to a} {f(x)}
            (r'\bpartial\s*\(([a-zA-Z0-9+\-*/^ ]+),\s*([a-zA-Z])\)', r'\\frac{\\partial}{\\partial \2} {\1}'),  # partial(f(x), x) -> \frac{\partial}{\partial x} {f(x)}
        ]

        latex_text = input_text
        for pattern, replacement in transformations:
            latex_text = re.sub(pattern, replacement, latex_text)
        latex_text = f"\\begin{{equation}}\n{latex_text}\n\\end{{equation}}"
        return latex_text
    except Exception as e:
        print("Ошибка при преобразовании текста в LaTeX:", str(e))
        return None
    
def latex_to_text(input_latex):
    """
    Преобразует LaTeX формулы в обычный текст.

    :param input_latex: Текст в формате LaTeX
    :return: Обычный текст с формулами
    """
    try:
        transformations = [
            (r'\\frac\{([a-zA-Z0-9+\-*/^ ]+)\}\{([a-zA-Z0-9+\-*/^ ]+)\}', r'\1/\2'),  # \frac{2}{3} -> 2/3
            (r'\\int\s*\{([a-zA-Z0-9+\-*/^ ]+)\}\s*\\,?\s*d([a-zA-Z])', r'integral(\1) d\2'),  # \int {f(x)} \, dx -> integral(f(x)) dx
            (r'\\sum\_\{([a-zA-Z0-9]+)=([a-zA-Z0-9]+)\}\^\{([a-zA-Z0-9]+)\}\s*\{([a-zA-Z0-9+\-*/^ ]+)\}', 
             r'sum(\4, \2=\3..\1)'),  # \sum_{i=1}^{n} {f(x)} -> sum(f(x), i=1..n)
            (r'\\lim\_\{([a-zA-Z0-9]+)\s*\\to\s*([a-zA-Z0-9]+)\}\s*\{([a-zA-Z0-9+\-*/^ ]+)\}', 
             r'lim(\3, \1->\2)'),  # \lim_{x \to a} {f(x)} -> lim(f(x), x->a)
            (r'\\frac\{\\partial\}\{\\partial\s*([a-zA-Z])\}\s*\{([a-zA-Z0-9+\-*/^ ]+)\}', 
             r'partial(\2, \1)'),  # \frac{\partial}{\partial x} {f(x)} -> partial(f(x), x)
            (r'\\sqrt\{([a-zA-Z0-9+\-*/^ ]+)\}', r'sqrt(\1)'),  # \sqrt{x} -> sqrt(x)
            (r'([a-zA-Z0-9]+)\s*\\cdot\s*([a-zA-Z0-9]+)', r'\1*\2'),  # x \cdot y -> x * y
            (r'([a-zA-Z]+)\^2', r'\1^2'),  # x^2 -> x^2
            (r'([a-zA-Z]+)\s*\_\s*([a-zA-Z0-9]+)', r'\1_\2'),  # x_1 -> x_1
            (r'([a-zA-Z]+)\s*=\s*([a-zA-Z0-9+\-*/^ ]+)', r'\1 = \2'),  # x = y -> x = y
        ]
        
        text = input_latex
        for pattern, replacement in transformations:
            text = re.sub(pattern, replacement, text)
        
        # Удаляем LaTeX-среду для уравнений
        text = re.sub(r'\\begin\{equation\}\n*', '', text)
        text = re.sub(r'\n*\\end\{equation\}', '', text)
        
        return text
    except Exception as e:
        print("Ошибка при преобразовании LaTeX в текст:", str(e))
        return None


def index(request):
    return render(request, 'main/index.html')

def phototolatex(request):
    latex_text = None
    image_url = None
    if request.method == "POST":
        form = PhotoInputForm(request.POST, request.FILES)
        if form.is_valid():
            if 'image' in request.FILES:
                image = request.FILES['image']
                fs = FileSystemStorage()
                filename = fs.save(image.name, image)
                image_url = fs.url(filename)
                file_path = os.path.join(fs.location, filename)
                latex_text = extract_latex_from_image(file_path)
    else:
        form = PhotoInputForm()
    return render(request, 'main/phototolatex.html', {'form': form, 'latex_text': latex_text, 'image_url': image_url})

def texttolatex(request):
    text = None
    if request.method == "POST":
        form = TextInputForm(request.POST)
        if form.is_valid():
            text = text_to_latex(form.cleaned_data['text'])
    else:
        form = TextInputForm()
    return render(request, 'main/texttolatex.html', {'form': form, 'text': text})

def team(request):
    return render(request,'main/team.html')

def antiplagiat(request):
    latex_text = None
    image_url = None
    text = None
    results = None
    # Обработка формы для загрузки изображения
    if request.method == "POST" and 'submit_image' in request.POST:
        form = PhotoInputForm(request.POST, request.FILES)
        if form.is_valid():
            if 'image' in request.FILES:
                image = request.FILES['image']
                fs = FileSystemStorage()
                filename = fs.save(image.name, image)
                image_url = fs.url(filename)
                file_path = os.path.join(fs.location, filename)
                latex_text = extract_latex_from_image(file_path)
                connection = connect_to_db()
                results = check_formula_uniqueness(latex_text, connection)
                if connection:
                    connection.close()
                if latex_text:
                    save_formula_to_db(latex_text)
    else:
        form = PhotoInputForm()

    # Обработка формы для ввода текста
    text = None
    if request.method == "POST":
        form1 = TextInputForm(request.POST)
        if form1.is_valid():
            text = text_to_latex(form1.cleaned_data['text'])
            connection = connect_to_db()
            results = check_formula_uniqueness(text, connection)
            if connection:
                connection.close()
            if text:
                save_formula_to_db(text)
    else:
        form1 = TextInputForm()

    return render(request, 'main/antiplagiat.html', {
        'form': form,
        'form1': form1,
        'latex_text': latex_text,
        'image_url': image_url,
        'text': text,
        'results': results
    })

def latextotext(request):
    text = None
    if request.method == "POST":
        form = TextInputForm(request.POST)
        if form.is_valid():
            text = latex_to_text(form.cleaned_data['text'])
    else:
        form = TextInputForm()
    return render(request, 'main/latextotext.html', {'form': form, 'text': text})