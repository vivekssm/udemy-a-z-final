
#same question from polynomial regressor


#IMPORTANT POINT
#svr is a type of regression similar to linear regression which set a max error and consider 
#only those point whose errors are less than max error otherwise neglect it.see graph,
#ceo salary is neglected


#svr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values.reshape(-1,1)#updated according to warning

#FEATURE SCALING IS IMPORTANT HERE BECAUSE LIBRARY USED DONT INCLUDE IT. IN PREVIOUS REGRESSOR
#LIBRARIES(POLY LINEAR) IT WAS ALREADY THERE IN THE MODULE.

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Fitting the Regression Model to the dataset
from sklearn.svm import SVR   #must read about this library
regressor=SVR(kernel="rbf")  #kernel is linear sigmoid poly and gaussian i.e rbf
                         #our problem is non-linear so select either polynomial or rbf
regressor.fit(X,y.ravel()) #ravel eeror me aya tha dalme ko,execute krne pe

# 

# Predicting a new result
         #scale 6.5 to fit it to current scale.
         #array is created because it is demand of library used here i.e, fit_transform.
y_pred =regressor.predict(sc_X.fit_transform((np.array([6.5])).reshape(-1,1)))
         #now defeature it to get to readable scale
         #1,31k that is less than 160 k.
y_pred=sc_y.inverse_transform(y_pred)



# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue') #blue line is svr
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#IMPORTANT POINT
#svr is a type of regression similar to linear regression which set a max error and consider 
#only those point whose errors are less than max error otherwise neglect it.here, ceo is neglected.
#see graph





# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (svr)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()