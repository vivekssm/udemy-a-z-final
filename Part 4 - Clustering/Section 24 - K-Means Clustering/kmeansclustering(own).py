#here we are trying to group people according to their income and expenditure



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4 ]].values

#using elbow method to decide no. of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("no. of clusters")
plt.ylabel("wcss")
plt.show()    #from graph it is seen that 5 cluster is perfect
                      
#applying kmeans to dataset
kmeans=KMeans(n_clusters=5,init="k-means++",max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)


#visualise tge results(code for 2-dimension datasets)
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c="red",label="careful")
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c="blue",label="standard")
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c="pink",label="target")
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c="green",label="careless")
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c="yellow",label="sensible")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c="magenta",label="centroid")
plt.title("cluster of clients")
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()