#here is an employ who is claimimg that i was regional manager in a company from 2 years 
#(level 6.5)and 
#my salary was $160000. the dataset contain info of salary of employees where he was working
#earlier. we r trying to find out wheather he was telling correct or not.

#DATA PREPROCESSING

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv("Position_Salaries.csv") 
x=dataset.iloc[:,1].values #1st colum i.e level is independent variable
x=dataset.iloc[:,1:2].values #it is done just to make our data in array and not vactor output same as above line
y=dataset.iloc[:,2].values #dependent variable


#here dataset is small so no test and training set

#HERE LET US CREATE LINEAR AND POLYNOMIAL REGRESSION BOTH TO COMPARE
#linear regression
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)

#polynomial regression
            #square the terms
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)  #here see x_poly it has 3 colums 2nd column has levels
                         #3rd column has square of 2nd level
                         #1st column has all ones like we added in our model previously. here model automatically add it

            #apply lineear regression to squared terms
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)


#visualise linear regression results
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg.predict(x),color="blue")
plt.xlabel("position")
plt.ylabel("salary")
plt.title("graph linear")

#visualise polynomials regression results
plt.scatter(x,y,color="red")
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color="blue") #line 34 but this code is open for all x to plot graph
plt.xlabel("position")
plt.xlabel("salary")
plt.title("graph polynomial")


#let us see with 3 degrees
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y,color="red")
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color="blue") 
plt.xlabel("position")
plt.xlabel("salary")
plt.title("graph polynomial")

#we keep trying. we can automate it using for loop

#4 degrees(it is perfect)

poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_poly,y)

#to make grid in the graph and make the graph more continuos we iterate it on smaller values
from sklearn.preprocessing import PolynomialFeatures
x_grid=np.arange(min(x),max(x),0.1) #added line
x_grid=x_grid.reshape(len(x_grid),1)#added line
plt.scatter(x,y,color="red")
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color="blue") #modified
plt.xlabel("position")
plt.xlabel("salary")
plt.title("graph polynomial")


#predicting new result on linear regression
lin_reg.predict(6.5)   #wrong

##predicting new result on polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5)) #it give outpot 158K so he was correct.