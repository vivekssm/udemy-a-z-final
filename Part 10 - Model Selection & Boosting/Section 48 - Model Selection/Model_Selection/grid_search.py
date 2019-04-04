#improving model by choosing best hyperparameter(hyperparameter are those parameter which aren't learnt)
#same question as previous (social add)


# Grid Search

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0) #yaha aur v parameters hai jinko hmne pichhli bar default pe 
#chhod dia tha is bar hm check karenge ki un prameters ko change karke kya ham result change kar sakte.
#jin parameters ko hm niche check kar rahe like -"c", kernel and gamma wo yahi k hai
#SVC K sare parameters padhe.. kya hai unka use kya hai
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.grid_search import GridSearchCV      #new version me model_selection me hai
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#upar wale parameters k probable values hmne input kia taki hame grid search se pata chale kya best hai
#yaha hmne 2 models ko try karne dia hai linear and rbf
#c hota hai overfitting control karne k lie default c=1 hota .. hm c ka value badhaenge to wo jyada control karega 
#overfitting ko but we will come with new problem underfitting
#gamma sirf non-linear k lie hota,default=0.5 hota hai   
         
grid_search = GridSearchCV(estimator = classifier, #because we have to classify data
                           param_grid = parameters, #hame parameters check krna hai.. upar wale line ka variable name hai parameters.. jisme hmne sara input kia hai
                           scoring = 'accuracy',    #here it can be accuracy,recall ,precision.. hame yaha accuracy k basis pe select krna
                           cv = 10,  #hmne train set ko divide kar dia 10 parts me i.e 10 fold cross_validation is going to be applied
                           n_jobs = -1) #add it when ju have large data set so that it can use all power of CPU
#
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_  #mean of accuracy of 10 folds
best_parameters = grid_search.best_params_  #gives list of best prameters seached by model 
#go in variable explorer to see ur results it has selected c=1,kernel="rbf",gamma=0.7

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()