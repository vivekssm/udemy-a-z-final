# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 17:59:21 2018

@author: DELL
"""
#decision tree(here also question same as polynomial regression..... check salary for level 6.5 
#it is 160k or not

#theory of this model (study from copy)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#import and apply library
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)   #leave it in default parameters
regressor.fit(x,y)

#now predict the salary according to model..
y_pred=regressor.predict(6.5)

# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue') #blue line is svr
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()          # this gives us continuous graph..so it isn't accurate graph..
                    #accurate graph should be stepwise. and continuous for interval

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (svr)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
