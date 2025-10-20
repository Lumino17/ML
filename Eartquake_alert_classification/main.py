import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# data preprocessing

df=pd.read_csv('earthquake_alert_balanced_dataset.csv')

def info(data):
    print(data.isnull().sum())
    print(data.columns)
    print(data.head())

def outliers(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    low_bound = q1 - 1.5*iqr
    up_bound = q3 + 1.5*iqr
    filtered_data = data[(data[col] >= low_bound) & (data[col] <= up_bound)]
    return filtered_data

def remove_outliers_df(data, cols):
    filtered = data.copy()
    for c in cols:
        filtered = outliers(filtered, c)
    return filtered

def scaling(data,cols):
    scaler = StandardScaler()
    data[cols] = scaler.fit_transform(data[cols])
    return data

# Main

df['alert'] = df['alert'].map({'green':0, 'yellow':1, 'orange':2, 'red':3})
col = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']

no_outliers_df = remove_outliers_df(df, col)
cleaned_df = scaling(no_outliers_df, col)

X=cleaned_df.drop('alert', axis=1)
y=cleaned_df['alert']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model implementations

def knn():
    knn_model = KNeighborsClassifier(n_neighbors=3)    #number of neigbours set to 3 after manual testing
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def naive_bayes():
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred = nb_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def svm():
    svm_model = SVC(kernel='rbf')  #using rbf(radial basis function) kernel after manual testing
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

svm()
knn()
naive_bayes()



  