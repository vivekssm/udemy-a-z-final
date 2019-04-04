# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 18:42:43 2018

@author: DELL
"""

#DATA PREPROCESSING

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv("50_Startups.csv") 
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values


#Encode state
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:,3]=labelencoder_x.fit_transform(x[:,3])
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()


#avoid dummy variable trap
x=x[:,1:]    #3 dummy variables reduced to 2

#splitting data into training test and cross-validation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)


#MULTIPLE LINEAR RFEGRESSION
#now apply the model on training set,, data is processed
from sklearn.linear_model import LinearRegression #import library
regressor=LinearRegression()
regressor.fit(x_train,y_train)


#now model has learnt, now let us predict y for test set
y_pred=regressor.predict(x_test)   #predicted y

#backward elimination
#(preperation of back elimination)
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1) #add column of ones to the matrix 
                                                  #to make it b0*x0 + b1*x1...+ bn*xn
              #(this was done in coursera video also.. for more study there..
              #here it is said thet library requires ones for calculation.
              # this addition makes matrix multiplication possible for the above equation.. earlier due to constant bo matrix multiplication was not possible..
            #here in x 1st column is ones, 2nd and 3rd are dummy variables and then rest
#(applying back elimination)
x_opt=x[:,[0,1,2,3,4,5]]  #xopt will contain essential parameters. initially it contains all variables 
                   #then we will eliminate them one by one
                   
                   #here column 0 contain all 1s
                   #column 1 and 2 contain dummy variables
                   #column 3 is r&d
                   #column 4 is administration..
                   #column 5 is marketting

                   #now move to pdf and read theory of back elimination.
                   #we select sl=0.05  now apply step 2 of pdf
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
                   #again repeat after removing parameters
x_opt=x[:,[0,1,3,4,5]]   #2nd is removed as it has highest p value
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

#again repeat
x_opt=x[:,[0,3,4,5]]   
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3,5]]     
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()#this is perfect model.. if p value and adjusted r-squared value both
                       #are taken care..(see theory from copy- last pages of regression..)


x_opt=x[:,[0,3]]     #if only p value is taken care remove 5 as it greater than 0.05
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()    
#AT THIS STAGE BACKPROP IS COMPLETE AND NOW ALL THE VARIABLES HAS VALUE LESS THAN SL VALUE I.E. 
#SL=0.05