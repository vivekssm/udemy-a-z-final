# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras


#here we have a data set of bank customers and we have to predict wheather customer will
#with the bank or exit(leave) ..

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  #3-12 ko hamne select kia kyuki iska impact hai result ko affect karne me so this is i/p
y = dataset.iloc[:, 13].values    #o/p jise hame predict krna

# Encoding categorical data (dummy variables will be created here country-france,spain,germany: gender-male and female)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])  ##country ko encode
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])   ##gender ko encode
##pichhe likha niche ka explanation (avoid dummy variable trap)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
###from sklearn.model_selection import train_test_split
#####X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
#####import tensorflow
from keras.models import Sequential #initialize NN
from keras.layers import Dense   #create layers

# Initialising the ANN(classifier me hi hum sara layer add karenge one by one)
##(rectifieer activation function for inner layers and sigmoid for outer layers)
classifier = Sequential()

# Adding the input layer and the first hidden layer (#relu is the fuction in this layer
#    (yaha avg lia 11+1/2=6                       #input dim is no of nodes in previous layer
#                                                  yaha 11 isle kyoki 11 independent variable hai
#                                                   6 agle layer me node
#                                                  init uniform islie kyoki hm chahte start small ho ctivation aur weight))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
     #
# Adding the second hidden layer(here no input_dim because it knows the no of nodes in previous layer)
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer (here output layer is only 1 because we need to have output so one node needed)
                       #(sigmoid fuction gives output 0 or 1 so sigmoid in output layer)
                       #if output has more than three categories then output dim=no of categories
                       #and activation =softmax but here 
                       #only two class that is customer either continue or leave the bank. so sigmoid is ok
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN (compiling means adding stocastic gradient descent to whole ANN)
            #adam is for selecting optimizer i.e. stocastic gradient descent in this case..
             #loss fuction is for selecting the way we will calculate loss fuction. in regression we calculate it
          #by squaring the difference of actual and predicted value but it isn't so in this case
                #here loss is calculated by log loss fuction(refer coursera copy for this.) 
           #if categories for output are just 2 then fuction is binary_crossentropy. if more then 
            #categorical_crossentropy..
         #here accuracy will increase slowly in ann and we have put it in box because the argument demands
          #list.. weights will be updated to increase the accuracy.. we are basically using
          #accuracy as a criterion to update weights.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
      #batch size ka meaning samjhaya tha coursera gradient descent me.
      #bada training set me divide kar lete 10-10 ka size to uddate the weights,
      #yaha 10 example k bad update hoga weight..
      #refer a-z copy..
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test) # (gives probability of customer to leave the bank.)
y_pred = (y_pred > 0.5)   #here threshold is taken very high in medical cases.

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)