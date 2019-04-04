#Here the dataset contains list of customer and products they purchased in the mall...
#so, using associative rule learning we will associate the products and recommend products to users
#using that analysis..


# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
               #transactions here will contain all the products in the dataset..
               #it contain 7500 list. one list for each customer
               
               
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
           #min_support=minimum support is selected (as per steps in copy)
             #(suppose a product is purchased 3 times a day so 21 times a week.)
             #(21/7500=0.003)
             
             
           #min_confidence=20% customers purchase both product
           
#visualise result
results=list(rules)   
#results_list = []
#for i in range(0, len(results)):
#   results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))
    #here variable results_list contain all the relevant pairs sorted with their decreasing importance
    #open the list to see light cfream and chicken is most important pair
    #so if people buy light cream.then they are likely to buy chicken..






