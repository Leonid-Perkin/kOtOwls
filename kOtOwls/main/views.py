from django.shortcuts import render
from django.http import HttpResponse
from .forms import TextInputForm

# Create your views here.
def index(request):
    text = None
    latex_text = None  # Переменная для текста "Latex Формула"
    image_url = None
    if request.method == "POST":
        form = TextInputForm(request.POST, request.FILES)  # Учитываем загружаемые файлы
        if form.is_valid():
            text = form.cleaned_data['text']
            if form.cleaned_data['image']:
                image_url = form.cleaned_data['image'].url  # Получаем URL изображения
            latex_text = "Latex Формула"  # Устанавливаем текст, когда нажата кнопка
    else:
        form = TextInputForm()

    return render(request, 'main/index.html', {'form': form, 'text': text, 'latex_text': latex_text, 'image_url': image_url})

