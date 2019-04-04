

import pandas as pd
 
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
n = len(dataset)
transactions = []
for i in range(0, n):
    transaction = []
    m = len(dataset.values[i])
    for j in range(0, m):
        data = str(dataset.values[i,j])
        if data != "nan":
            transaction.append(data)
    transactions.append(transaction)
 
results = []
from fim import eclat
rules = eclat(tracts = transactions, zmin = 1)
rules.sort(key = lambda x: x[1], reverse = True)