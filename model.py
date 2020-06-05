import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("./data/titanic.csv")
predictors = ['Age', 'Sex','Pclass',  'SibSp', 'Parch', 'Fare', 'Embarked']
label = 'Survived'


df["Sex"]=np.where(df["Sex"]=="male",0,1)
df["Embarked"]=np.where(df["Embarked"]=="S",0,
                                  np.where(df["Embarked"]=="C",1,
                                           np.where(df["Embarked"]=="Q",2,3)
                                          )
                                 )
df_train, df_test, y_train, y_test = train_test_split(df[predictors], df[label], test_size=0.20, random_state=11)

age_fillna = df_train.Age.mean()
embarked_fillna = df_train.Embarked.value_counts().index[0]


df_train.Age = df_train.Age.fillna(df.Age.mean())
df_train.Embarked = df_train.Embarked.fillna(embarked_fillna)

df_test.Age = df_test.Age.fillna(df.Age.mean())
df_test.Embarked = df_test.Embarked.fillna(embarked_fillna)

classifier = GaussianNB()
classifier.fit(df_train, y_train)

y_pred = classifier.predict(X=df_test)
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred))

import joblib
joblib.dump(classifier, './model.pkl')

clf = joblib.load( './model.pkl')

to_predict = [28, 1, 1,1, 2,48, 1 ]
to_predict = np.array(to_predict).reshape(1, -1)
print(clf.predict_proba(to_predict))