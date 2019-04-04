#DATA PREPROCESSING

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv("Data.csv") #jis python file me kam kar rahe aur jaha data hai dono same folder me save hona chahie
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0) #Imputer google karne pe pata
                                                    #chalega (bracket) me kya  kyu dalna
                                                #similarly to all brackets in this program
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])



#CATEGORICAL DATA(dummy variables craeted for countries and purchased as they are in words.)
              #when countries are to be coded as 0,1,2,3..
from sklearn.preprocessing import LabelEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(x[:,0])
             #when to be coded as 3 column matrix where country having it is 1 and rest are 0
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])#onehotencoder google kare here 
                                                #categorical_features is
                                              #0 beacuse we want to encode 1st column of data 
x=onehotencoder.fit_transform(x).toarray() #x me one hot encoder apply karna 1st column upar
                                           #line se  clear


#splitting data into training test and cross-validation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0) #google 
            #train_test_split to understand meaning of all the terms in the bracket. 20% of data is 
            #is test set.


#FEATURE SCALING(we do it to put the variables into same scale here age is in tens and salary
                 #in thousands so salary may dominate in maths. so put them to same scale )
                 #(two types of scaling are standrization and normalization)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)#pahle standard scaler ko fit karo training set me aur tab tranform
x_test=sc_x.transform(x_test)#pahle se fit hai sirf transform karo.fit sirf ek jagah kar raha kyu 
              #because we want both to be scaled on same scale.
              #here we dont need to scale y because it doesn't vary much.

              #one question often asked is should we scale dummy variables?(like countries and purchased)                           
              #answer is depends on sitiation
              #here we have scaled countries(i.e x) and not purchased (i.e y)
              
