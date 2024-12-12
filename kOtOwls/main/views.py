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

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger("pydantic").setLevel(logging.CRITICAL)

def extract_latex_from_image(image_path):
    """Извлечение формулы в формате LaTeX с изображения с помощью pix2tex."""
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
    

def index(request):
    pass
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
            text = form.cleaned_data['text']
    else:
        form = TextInputForm()
    return render(request, 'main/texttolatex.html', {'form': form, 'text': text})