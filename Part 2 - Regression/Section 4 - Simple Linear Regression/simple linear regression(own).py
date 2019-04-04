
#SIMPLE LINEAR REGRESSION
#DATA PREPROCESSING

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv("Salary_Data.csv") 
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#splitting data into training test and cross-validation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=1/3,random_state=0)

#simple linear regression library will take care of scaling

#now apply the model data is processed
from sklearn.linear_model import LinearRegression #import library
regressor=LinearRegression()
regressor.fit(x_train,y_train)   #make the model learn on training set



#now model has learnt, now let us predict y for test set
y_pred=regressor.predict(x_test)   #predicted y

#visualize train data and regression line
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue") #regression line
plt.xlabel("experience")
plt.ylabel("salary")
plt.title("xtrain graph")
plt.show()

#visualize test data and regression line
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")  #we are training on train set so we will keep it same.
plt.xlabel("experience")
plt.ylabel("salary")
plt.title("xtest graph")
plt.show()