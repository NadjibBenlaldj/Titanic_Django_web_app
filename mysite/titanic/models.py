from django.db import models
import numpy as np

class Passenger(models.Model):

    name = models.CharField(max_length=50,null=True)
    sex = models.PositiveSmallIntegerField(null=True)
    sibsp = models.FloatField(null=True)
    parch = models.FloatField(null=True)
    fare = models.FloatField(null=True)
    age = models.FloatField(null=True)
    ticket_class = models.PositiveSmallIntegerField(null=True)
    embarked = models.PositiveSmallIntegerField(null=True)

    def __str__(self):
        return self.name

    def survival_proba(self, classifier):

        to_predict = [self.age, self.sex, self.ticket_class,self.sibsp , self.parch,self.fare, self.embarked ]
        to_predict = np.array(to_predict).reshape(1, -1)
        proba=  classifier.predict_proba(to_predict)
        return(proba[0][0])
