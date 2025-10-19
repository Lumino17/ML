import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("House_Rent_Dataset.csv")
print(df.head())
print(df.info())
new_df=df.drop(['Posted On','Floor','Area Type','Area Locality','City','Furnishing Status','Tenant Preferred','Point of Contact'],axis=1)
print(new_df.head())

X=new_df[['BHK','Size','Bathroom']]
Y=new_df['Rent']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

model=LinearRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)

mse=mean_squared_error(y_test,pred)
r2=r2_score(y_test,pred)

print("mse:",mse)
print("r2:",r2)



