import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

df=pd.read_csv('predictive_maintenance.csv')

def info(data):
    print(data.isnull().sum())
    print(data.dtypes)
    print(data.columns)
    print(data.head())

df['Type']= df['Type'].map({'L':0,'M':1,'H':2})
X=df[['Type','Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']]
Y=df['Target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
model = LogisticRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
print("Accuracy:",accuracy_score(Y_test, y_pred))

