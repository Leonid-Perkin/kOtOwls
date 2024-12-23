from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('phototolatex/', views.phototolatex),
    path('texttolatex/', views.texttolatex),
    path('team/',views.team),
    path('antiplagiat/',views.antiplagiat),
    path('latextotext/',views.latextotext),
    path('doc/',views.doc),
    path('formulaeditor/',views.formulaeditor),
    path('save/', views.save_formula, name='save_formula'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)