# views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import *

def index(request):
    text = None
    latex_text = None 
    image_url = None

    if request.method == "POST":
        form = TextInputForm(request.POST, request.FILES)
        if form.is_valid():
            text = form.cleaned_data['text']
            if 'image' in request.FILES:
                image = request.FILES['image']
                fs = FileSystemStorage()
                filename = fs.save(image.name, image)
                image_url = fs.url(filename)

            latex_text = "Latex Формула"
    else:
        form = TextInputForm()

    return render(request, 'main/index.html', {'form': form, 'text': text, 'latex_text': latex_text, 'image_url': image_url})

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
            latex_text = "Latex Формула"
    else:
        form = PhotoInputForm()
    return render(request, 'main/phototolatex.html', {'form': form, 'latex_text': latex_text, 'image_url': image_url})