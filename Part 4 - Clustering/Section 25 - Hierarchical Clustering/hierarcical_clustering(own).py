#here we are trying to group people according to their income and expenditure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4 ]].values

#using dendrogram method to decide optimum no. of clusters
import scipy.cluster.hierarchy as sch
dendroid=sch.dendrogram(sch.linkage(x,method="ward"))
plt.title("dendrogram")
plt.xlabel("customer")
plt.ylabel("euclidian distance")
plt.show()    #green line from 100 to 250 is the longest line without passing any horizontal line.
#put a horizontal line perpendicular to that.
#from graph it is seen that 5 cluster is perfect
                      
#applying hierchial clustering  to dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity="euclidean",linkage="ward")
y_hc=hc.fit_predict(x)

#visualise tge results(code for 2-dimension datasets)
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,c="red",label="careful")
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,c="blue",label="standard")
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,c="pink",label="target")
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,c="green",label="careless")
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,c="yellow",label="sensible")
plt.title("cluster of clients")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()