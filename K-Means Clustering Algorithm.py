import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\OneDrive\Work\Projects\K-Means Clustering Algorithm\iris.csv")

feature = df.iloc[:, [0,1,2,3]].values

kmeans5 = KMeans(n_clusters=5) #Arbitrarily selected 5 for first iteration
label_kmeans5 = kmeans5.fit_predict(feature)
print(label_kmeans5)

#Implementing Elbow graph
Error = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(feature)
    kmeans.fit(feature)
    Error.append(kmeans.inertia_)
plt.plot(range(1, 11), Error)
plt.title('Elbow Graph')
plt.xlabel('No. of clusters')
plt.ylabel('Error')
plt.show()

kmeans3 = KMeans(n_clusters=3) #Selected 3 for second iteration based on Elbow graph
label_kmeans3 = kmeans3.fit_predict(feature)
print(label_kmeans3)

plt.scatter(feature[:, 0], feature[:, 1], c=label_kmeans3, cmap = 'brg')
plt.show()