# forms.py
from django import forms

class TextInputForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={'placeholder': 'Введите ваш текст здесь...'}))
class PhotoInputForm(forms.Form):
    image = forms.ImageField()
    
class AntiPlagiarismForm(forms.Form):
    formula = forms.CharField(widget=forms.Textarea(attrs={'placeholder': 'Введите вашу формулу'}), label='Формула')
    image = forms.ImageField(required=False, label='Загрузите изображение', widget=forms.ClearableFileInput(attrs={'multiple': True}))
    submit = forms.CharField(widget=forms.HiddenInput(), initial='Отправить')  # Это поле скрыто для кнопки