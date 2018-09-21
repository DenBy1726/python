#normalise and standartise
from sklearn import preprocessing
import matplotlib.pyplot as plt
from numpy import random, array
import numpy as np

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
X = data[:,0]
Y = data[:,1]

plt.scatter(X,Y)
plt.title("without scaling")
plt.savefig("./data mining/plot/normalisation/without_normalisation.png",format="png")

plt.close()

# normalize the data attributes
normalized_X = preprocessing.normalize([X])
normalized_Y = preprocessing.normalize([Y])

plt.scatter(normalized_X,normalized_Y)
plt.title("normalisation")
plt.savefig("./data mining/plot/normalisation/normaliastion.png",format="png")
# standardize the data attributes
standardized_X = preprocessing.scale(X)
standardized_Y = preprocessing.scale(Y)

plt.close()

plt.scatter(standardized_X,standardized_Y)
plt.title("standartisation")
plt.savefig("./data mining/plot/normalisation/standartisation.png",format="png")

plt.close()

plt.scatter(X,Y)
plt.scatter(normalized_X,normalized_Y)
plt.scatter(standardized_X,standardized_Y)

plt.legend(["without", "normaliastion", "standartisation"])
plt.savefig("./data mining/plot/normalisation/test.png",format="png")

plt.close()

plt.scatter(normalized_X,normalized_Y)
plt.scatter(standardized_X,standardized_Y)

plt.legend(["normaliastion", "standartisation"])
plt.savefig("./data mining/plot/normalisation/test2.png",format="png")
