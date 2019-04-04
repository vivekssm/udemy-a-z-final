# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:18:22 2018

@author: DELL
"""
#DATA PREPROCESSING

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing datasets
dataset=pd.read_csv("Data.csv") 
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#splitting data into training test and cross-validation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

#FEATURE SCALING
"""from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""