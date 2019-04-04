#IN THIS PROBLEM WE HAVE TO CLASSIFY THE PEOPLE ACCORDING TO WHO  BOUGHT THE car AND WHO DIDN'T
#ACCORDING TO AGE AND INCOME(SEE DATASET IN MS-EXCEL)




# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting classifier to the Training set
   # Create your classifier here
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)   #THIS MATRIX WILL TELL US HOW MANY CORRECT AND INCORRECT
                     #PREDICTIONS DID WE MAKE.  HERE WE MADE 65+24 CORRECT PREDICTIONS 
                     #AND 8+3 INCORRECT PREDICTIONS..

# Visualising the Training set results
                     #it is 3-d graph 
                     #x-axis is age; y-axis is salary
                     #points tell who bought and who didn't bought the car
                     #here green points are those who actually bought the car and red are
                     #those who didn't... the line is our classifier(prediction boundry)
                     #it is straight because logistic regression is linear classifier
                     
                     #in 3-d it will be straight plane..
                     
                     
                     
                     #from the graph we will see
                     #higher age people buy it(green points are more in right of x-axis)
                     #low age people with high salary buy it(green points are there in top of red region)
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
                    
                #let us scatter the whole plot from min to max with resolution of 0.01
x1,x2=np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
               
                #below line divides the graph into two parts,predict for each point and colour 
                #that pixel into red or green
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
                
               #below code give min max  limits of graph to be plotted

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

              #this code plot the graph :plt.sctter is used earlier also
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)


plt.title('Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()     #used for index on right hand top of graph
plt.show()

                     
                     
                     
                     
# Visualising the Test set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1,x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()