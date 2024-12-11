# views.py
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import TextInputForm

def index(request):
    text = None
    latex_text = None  # Переменная для текста "Latex Формула"
    image_url = None

    if request.method == "POST":
        form = TextInputForm(request.POST, request.FILES)  # Учитываем загружаемые файлы
        if form.is_valid():
            text = form.cleaned_data['text']
            
            # Обрабатываем загруженное изображение
            if form.cleaned_data['image']:
                image = form.cleaned_data['image']
                fs = FileSystemStorage()  # Создаем объект FileSystemStorage для сохранения
                filename = fs.save(image.name, image)  # Сохраняем изображение
                image_url = fs.url(filename)  # Получаем URL изображения

            latex_text = "Latex Формула"  # Устанавливаем текст, когда нажата кнопка
    else:
        form = TextInputForm()

    return render(request, 'main/index.html', {'form': form, 'text': text, 'latex_text': latex_text, 'image_url': image_url})
