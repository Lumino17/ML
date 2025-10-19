import pandas as pd 

df=pd.read_csv('car_sales_data.csv')

def info():
    print(df.info())
    print(df.isnull().sum())
    print(df.head())

print(info())