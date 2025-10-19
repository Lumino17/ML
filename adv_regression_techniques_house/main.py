import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('boston.csv')

def info(data):
    print(data.head())
    print(data.describe())
    print(data.isnull().sum())

def outliers(data, cols):
    q1 = data[cols].quantile(0.25)
    q3 = data[cols].quantile(0.75)
    iqr = q3 - q1
    low_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask = ((data[cols] < low_bound) | (data[cols] > upper_bound)).any(axis=1)
    new_df = data.loc[~mask].copy()
    return new_df

def normalize(data,cols):
    scaler = MinMaxScaler()
    data[cols] = scaler.fit_transform(data[cols])
    return data

def ridge_regression():
    model = Ridge(alpha=0.175)    # Choose alpha manual tuning
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("mse:", mse,"r2:", r2)  

def lasso_regression():
    model = Lasso(alpha=0.0004)   # Calculated using manual tuning
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("mse:", mse,"r2:", r2)

def elasticnet_regression():
    model = ElasticNet(alpha=0.0006, l1_ratio=0.72)  # Calculated using manual tuning
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("mse:", mse,"r2:", r2)        

cols = df.columns
df_no_outliers = outliers(df, cols)
df_normalized = normalize(df_no_outliers, cols)

X=df_normalized[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
Y=df_normalized['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

ridge_regression()
lasso_regression()
elasticnet_regression()




