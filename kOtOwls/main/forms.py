# forms.py
from django import forms

class TextInputForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={'placeholder': 'Введите ваш текст здесь...'}))
    image = forms.ImageField(required=False)  # Добавляем поле для загрузки изображения
