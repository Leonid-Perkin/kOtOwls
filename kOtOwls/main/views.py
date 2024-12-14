# views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import *

import os
import torch
from PIL import Image
from pix2tex.cli import LatexOCR
import warnings
import logging

from sympy import symbols, latex, parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application
import re

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
    return render(request,'main/antiplagiat.html')