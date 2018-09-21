#issues
#https://datascience.stackexchange.com/questions/9793/running-examples-from-scikit-learn-tutorials
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy import random, array, float
import numpy as np
from scipy.spatial.distance import cdist

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range (k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([random.normal(incomeCentroid, 10000.0), random.normal(ageCentroid, 2.0)])
    X = array(X)
    return X

data = createClusteredData(100, 5)

model = KMeans(n_clusters=5)

# Note I'm scaling the data to normalize it! Important for good results.
scaller = StandardScaler().fit(data)
t_data = scaller.transform(data)
model = model.fit(t_data)

# And we'll visualize it:
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))

centroids = scaller.inverse_transform(model.cluster_centers_)

plt.scatter(centroids[:, 0], centroids[:, 1],c="r", marker="x")
plt.savefig("./data mining/plot/kmeannorm.png", format="png")
plt.close()

model = KMeans(n_clusters=5)

model.fit(data)

# And we'll visualize it:
plt.figure(figsize=(8, 6))
plt.scatter(data[:,0], data[:,1], c=model.labels_.astype(float))

centroids = model.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1],c="r", marker="x")
plt.savefig("./data mining/plot/kmean.png", format="png")

plt.close()

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(t_data)
    distortions.append(
        sum(
            np.min(
                cdist(data, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / t_data.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.savefig("./data mining/plot/kmeanelbow.png", format="png")
