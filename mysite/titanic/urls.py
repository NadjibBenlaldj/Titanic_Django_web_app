from django.urls import path

from . import views

app_name = 'titanic'
urlpatterns = [
    path('', views.index, name='index'),
    path('results', views.predict_proba, name='results'),
]