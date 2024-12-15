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
    """
    Establishes a connection to the SQLite database.

    Returns:
        sqlite3.Connection: A connection object to the SQLite database.
    """
    connection = sqlite3.connect('main/formulas.db')
    return connection

def save_formula_to_db(latex_text):
    """
    Save a LaTeX formula to the database with a default legend value of 'no'.
    Args:
        latex_text (str): The LaTeX formula to be saved.
    Returns:
        None
    """
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS formulas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            formula TEXT NOT NULL,
            legend TEXT
        )
    ''')
    cursor.execute('''
        INSERT INTO formulas (formula, legend)
        VALUES (?, ?)
    ''', (latex_text, 'no'))
    connection.commit()
    connection.close()
def fetch_formulas(connection):
    """
    Fetches formulas and their legends from the SQLite database.

    Args:
        connection (sqlite3.Connection): The connection object to the SQLite database.

    Returns:
        list of tuple: A list of tuples where each tuple contains a formula and its corresponding legend.
    """
    cursor = connection.cursor()
    cursor.execute("SELECT formula, legend FROM formulas")
    formulas = [(row[0], row[1]) for row in cursor.fetchall()]
    cursor.close()
    return formulas
def compare_formulas(formula1, formula2):
    """
    Compares two formulas and returns a similarity coefficient using difflib.

    Args:
        formula1 (str): The first formula to compare.
        formula2 (str): The second formula to compare.

    Returns:
        float: A similarity coefficient between 0 and 1, where 1 indicates identical formulas.
    """
    similarity = difflib.SequenceMatcher(None, formula1, formula2).ratio()
    return similarity
def neural_network_similarity(formula1, formula2):
    """
    Uses a neural network to compute the similarity between two formulas.

    Args:
        formula1 (str): The first formula to compare.
        formula2 (str): The second formula to compare.

    Returns:
        float: A similarity score between 0 and 1, where 1 indicates identical formulas and 0 indicates no similarity.
    """
    inputs = tokenizer([formula1, formula2], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        similarity_score = cosine_similarity(embeddings.cpu().numpy())[0][1]
    return similarity_score
def check_formula_uniqueness(input_formula, connection):
    """
    Check the uniqueness of the input formula by comparing it with existing formulas in the database.

    This function uses two methods to compare the input formula with existing formulas:
    1. A difflib-based comparison.
    2. A neural network-based comparison.

    Args:
        input_formula (str): The formula to be checked for uniqueness.
        connection (object): The database connection object.

    Returns:
        dict: A dictionary containing the following keys:
            - 'max_similarity_difflib' (float): The maximum similarity percentage found using difflib.
            - 'most_similar_formula_difflib' (str): The formula that is most similar to the input formula using difflib.
            - 'most_similar_legend_difflib' (str): The legend associated with the most similar formula using difflib.
            - 'max_similarity_nn' (float): The maximum similarity percentage found using the neural network.
            - 'most_similar_formula_nn' (str): The formula that is most similar to the input formula using the neural network.
            - 'most_similar_legend_nn' (str): The legend associated with the most similar formula using the neural network.
    """
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
    """
    Extracts LaTeX formula from an image using OCR.

    Args:
        image_path (str): The file path to the image from which to extract the LaTeX formula.

    Returns:
        str: The extracted LaTeX formula. If the formula does not start with "\\begin{array}", it will be wrapped in "\\begin{equation}" and "\\end{equation}".
        None: If an error occurs during the extraction process.

    Raises:
        Exception: If there is an error during the image processing or OCR.
    """
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

def text_to_latex(input_text):
    """
    Converts a given mathematical expression in plain text to LaTeX format.
    This function applies a series of regex transformations to convert common
    mathematical notations in plain text to their corresponding LaTeX representations.
    The resulting LaTeX code is wrapped in an equation environment.
    Args:
        input_text (str): The input mathematical expression in plain text.
    Returns:
        str: The converted LaTeX formatted string wrapped in an equation environment.
        None: If an error occurs during the conversion process.
    Example:
        >>> text_to_latex("2*x + sqrt(y) = 4")
        '\\begin{equation}\n2 \\cdot x + \\sqrt{y} = 4\n\\end{equation}'
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
    Converts a LaTeX string to a plain text representation by applying a series of regex transformations.
    Args:
        input_latex (str): The LaTeX string to be converted.
    Returns:
        str: The plain text representation of the LaTeX string, or None if an error occurs.
    Transformations:
        - \frac{a}{b} -> a/b
        - \int {f(x)} \, dx -> integral(f(x)) dx
        - \sum_{i=1}^{n} {f(x)} -> sum(f(x), i=1..n)
        - \lim_{x \to a} {f(x)} -> lim(f(x), x->a)
        - \frac{\partial}{\partial x} {f(x)} -> partial(f(x), x)
        - \sqrt{x} -> sqrt(x)
        - x \cdot y -> x * y
        - x^2 -> x^2
        - x_1 -> x_1
        - x = y -> x = y
    Note:
        The function also removes LaTeX equation environment delimiters (\begin{equation} and \end{equation}).
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
        text = re.sub(r'\\begin\{equation\}\n*', '', text)
        text = re.sub(r'\n*\\end\{equation\}', '', text)
        
        return text
    except Exception as e:
        print("Ошибка при преобразовании LaTeX в текст:", str(e))
        return None


def index(request):
    """
    Handle the request to the index page and render the 'main/index.html' template.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered 'main/index.html' template.
    """
    return render(request, 'main/index.html')

def phototolatex(request):
    """
    Handle the photo to LaTeX conversion request.

    This view handles both GET and POST requests. For GET requests, it initializes an empty form.
    For POST requests, it processes the uploaded image, saves it to the file system, extracts LaTeX
    text from the image, and returns the result along with the form and image URL.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response with the form, extracted LaTeX text, and image URL.
    """
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
    """
    Handle the text to LaTeX conversion form submission.

    This view function processes a form submission for converting text to LaTeX.
    If the request method is POST, it validates the form and converts the input text
    to LaTeX format. If the request method is GET, it initializes an empty form.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response with the form and the converted text (if any).
    """
    text = None
    if request.method == "POST":
        form = TextInputForm(request.POST)
        if form.is_valid():
            text = text_to_latex(form.cleaned_data['text'])
    else:
        form = TextInputForm()
    return render(request, 'main/texttolatex.html', {'form': form, 'text': text})

def team(request):
    """
    Handles the request to display the team page.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered team page.
    """
    return render(request,'main/team.html')

def antiplagiat(request):
    """
    Handles the anti-plagiarism functionality by processing image uploads and text inputs.
    This view supports two forms:
    1. PhotoInputForm: For uploading an image containing LaTeX formulas.
    2. TextInputForm: For entering text that will be converted to LaTeX.
    Depending on the form submitted, it extracts LaTeX from the image or converts text to LaTeX,
    checks the uniqueness of the formula in the database, and saves the formula if it is unique.
    Args:
        request (HttpRequest): The HTTP request object.
    Returns:
        HttpResponse: The rendered HTML page with the forms and results.
    """
    latex_text = None
    image_url = None
    text = None
    results = None
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
    """
    Handle the conversion of LaTeX input to plain text.

    This view function processes a POST request containing LaTeX input,
    converts it to plain text, and renders the result in a template.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML response with the form and converted text.

    Template:
        main/latextotext.html

    Context:
        form (TextInputForm): The form for inputting LaTeX text.
        text (str or None): The converted plain text, or None if the form is not valid or not submitted.
    """
    text = None
    if request.method == "POST":
        form = TextInputForm(request.POST)
        if form.is_valid():
            text = latex_to_text(form.cleaned_data['text'])
    else:
        form = TextInputForm()
    return render(request, 'main/latextotext.html', {'form': form, 'text': text})

def doc(request):
    """
    Handles the HTTP request for the documentation page.

    Args:
        request (HttpRequest): The HTTP request object.

    Returns:
        HttpResponse: The rendered HTML page for the documentation.
    """
    return render(request,'main/doc.html')