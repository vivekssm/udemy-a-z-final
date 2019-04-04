#Here the dataset contains list of customer and products they purchased in the mall...
#so, using associative rule learning we will associate the products and recommend products to users
#using that analysis..


# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv')
transactions = []                        #initiating a list
for i in range(0, 7500):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
  #it contain 7500 list. one list for each customer.20 is max no of product a customer has purchased according to dataset
   #bas hmne dataset ko list me convert kia apne use k lie            
               
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
           #min_support=minimum support is selected (as per steps in copy)
             #(if a product is purchased 3 times a day so 21 times a week... it is minimum times a product should be purchased)
             #(21/7500=0.003)
             #min_lenght=2(we need minimum 2 product in a basket)
             
             #min_lift ka meaning copy se
           #min_confidence=20% customers purchase both product
           
#visualise result
results=list(rules)   

#results khol k dekh la.. ye apne ap sorted hai.. top pe wo jo sabse jyada connected hai and decreasing then


#results_list = []
#for i in range(0, len(results)):
#    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))
    #here variable results_list contain all the relevant pairs sorted with their decreasing importance
    #open the list to see light cfream and chicken is most important pair
    #so if people buy light cream.then they are likely to buy chicken..






