import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

df=pd.read_csv('CollegePlacement.csv')

def info(data):
    print(data.isnull().sum())
    print(data.columns)
    print(data.head())

df[['Internship_Experience','Placement']] = df[['Internship_Experience','Placement']].replace({'Yes':1,'No':0})

X=df[['IQ','Prev_Sem_Result','CGPA','Academic_Performance','Internship_Experience','Extra_Curricular_Score','Communication_Skills','Projects_Completed']]
Y=df['Placement']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

def rfc():
    rfc_model = RandomForestClassifier(n_estimators=100,max_depth=4,random_state=42)
    rfc_model.fit(X_train, Y_train)
    Y_pred = rfc_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

def dtc():
    dtc_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    dtc_model.fit(X_train, Y_train)
    Y_pred = dtc_model.predict(X_test)
    print(classification_report(Y_test, Y_pred))

rfc()
dtc()