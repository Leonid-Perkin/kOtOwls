from django.shortcuts import render
from django.http import HttpResponse
from .forms import TextInputForm

# Create your views here.
def index(request):
    text = None
    if request.method == "POST":
        form = TextInputForm(request.POST)
        if form.is_valid():
            text = form.cleaned_data['text']
    else:
        form = TextInputForm()

    return render(request, 'main/index.html', {'form': form, 'text': text})

