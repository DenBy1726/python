from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from pylab import *
import matplotlib.pyplot as plt
from itertools import cycle

iris = load_iris()

numSamples, numFeatures = iris.data.shape
print(numSamples)
print(numFeatures)
print(list(iris.target_names))

#two component
X = iris.data
pca = PCA(n_components=2, whiten=True).fit(X)
print(pca.components_)
X_pca = pca.transform(X)

print(sum(pca.explained_variance_ratio_))

colors = cycle('rgb')
target_ids = range(len(iris.target_names))
plt.figure()
for i, c, label in zip(target_ids, colors, iris.target_names):
    plt.scatter(X_pca[iris.target == i, 0], X_pca[iris.target == i, 1] * len(X_pca[iris.target == i, 0]), c=c, label=label)
plt.legend()
plt.text(-1.5,125,"percentage is " + str(int(sum(pca.explained_variance_ratio_) * 100)))
plt.savefig("./data mining/plot/pca/2.png", format="png")
plt.close()


#one component
X = iris.data
pca = PCA(n_components=1, whiten=True).fit(X)
print(pca.components_)
X_pca = pca.transform(X)

print(sum(pca.explained_variance_ratio_))

colors = cycle('rgb')
target_ids = range(len(iris.target_names))
plt.figure()
for i, c, label in zip(target_ids, colors, iris.target_names):
    plt.scatter(X_pca[iris.target == i, 0], [0] * len(X_pca[iris.target == i, 0]), c=c, label=label)
plt.legend()
plt.text(-1.5,0.007,"percentage is " + str(int(sum(pca.explained_variance_ratio_) * 100)))
plt.savefig("./data mining/plot/pca/1.png", format="png")