# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 15:18:34 2018

@author: DELL
"""
#same question

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#import and apply library
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0) #no of trees=nestimators, rest  #leave it in default parameters
                         #change no of trees by changing n_estimatrs.
regressor.fit(x,y)

        #100 trees
"""regressor=RandomForestRegressor(n_estimators=100,random_state=0)
regressor.fit(x,y)"""
        #300 trees(perfect) run and see it gives exact 160k
"""""regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)"""""

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

