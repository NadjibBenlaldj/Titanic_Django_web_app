
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic
from django.utils import timezone

from .models import Passenger
import joblib

classifier = joblib.load('../model.pkl')

def index(request):

    return render(request, 'titanic/index.html')


def predict_proba(request):
    print(request)
    if request.method == 'POST':
        temp = Passenger()
        temp.name = request.POST.get('name')
        temp.sex = int(request.POST.get('sex'))
        temp.age = float(request.POST.get('age'))
        temp.sibsp= int(request.POST.get('sibsp'))
        temp.parch = int(request.POST.get('parch'))
        temp.fare = float(request.POST.get('fare'))
        temp.ticket_class = int(request.POST.get('ticket_class'))
        temp.embarked = int(request.POST.get('embarked'))

    temp.save()

    scoreval = temp.survival_proba(classifier)
    context = {'scoreval': round(scoreval*100), 'passenger': temp }
    return render(request, 'titanic/results.html',context)

