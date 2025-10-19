import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv('Student_Performance.csv')

def info(data):
    print(data.info())
    print(data.isnull().sum())
    print(data.head())

def scale_data(data):
    scaler=MinMaxScaler()
    normalized_data=scaler.fit_transform(data)
    return pd.DataFrame(normalized_data,columns=data.columns)

new_df = df.copy()
new_df['Extracurricular Activities'] = new_df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
new_df['Extracurricular Activities'] = new_df['Extracurricular Activities'].fillna(0).astype(int)

scaled_data=scale_data(new_df)

X=scaled_data[['Hours Studied','Previous Scores','Extracurricular Activities','Sleep Hours','Sample Question Papers Practiced']]
Y=scaled_data['Performance Index']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

model1=LinearRegression()
model2=RandomForestRegressor()
model3=XGBRegressor()

def LR():
    model1.fit(x_train,y_train)
    y_pred=model1.predict(x_test)
    print("Linear Regression Results:")
    print("MAE:", mean_absolute_error(y_test,y_pred))
    print("MSE:", mean_squared_error(y_test,y_pred))
    print("R2 Score:", r2_score(y_test,y_pred))

def RF():
    model2.fit(x_train,y_train)
    y_pred=model2.predict(x_test)
    print("Random Forest Regressor Results:")
    print("MAE:", mean_absolute_error(y_test,y_pred))
    print("MSE:", mean_squared_error(y_test,y_pred))
    print("R2 Score:", r2_score(y_test,y_pred))

def XGB():
    model3.fit(x_train,y_train)
    y_pred=model3.predict(x_test)
    print("XGBoost Regressor Results:")
    print("MAE:", mean_absolute_error(y_test,y_pred))
    print("MSE:", mean_squared_error(y_test,y_pred))
    print("R2 Score:", r2_score(y_test,y_pred)) 


LR()
RF()
XGB()

