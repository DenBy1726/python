import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#Create fake income/age clusters for N people in k clusters
def createClusteredData(N, k):
    pointsPerCluster = float(N)/k
    X = []
    y = []
    for i in range (k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)

    c = np.c_[X.reshape(len(X), -1), y.reshape(len(y), -1)]
    np.random.shuffle(c)

    a2 = c[:, :X.size//len(X)].reshape(X.shape)
    b2 = c[:, X.size//len(X):].reshape(y.shape)

    return a2, b2

from pylab import *

(data, index) = createClusteredData(150, 5)
testData = data[100:]
testIndex = index[100:]
data = data[:100]
index = index[:100]
scaller = StandardScaler().fit(data)
testScaller = StandardScaler().fit(testData)
data = scaller.transform(data)
testData = testScaller.transform(testData)

from sklearn import svm, datasets

C = 1
svc = svm.SVC(kernel='linear', C=C, max_iter=500).fit(data, index)

def plotPredictions(clf,name):
    xx, yy = np.meshgrid(np.arange(0,200000, 1000),np.arange(0, 100, 0.5))
    Z = clf.predict(scaller.transform(np.c_[xx.ravel(), yy.ravel()]))
    data_ = scaller.inverse_transform(data)

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    data_ =scaller.inverse_transform(data)
    plt.scatter(data_[:,0], data_[:,1], c=index.astype(np.float))

    percent = test(svc,data,index)
    percentTest = test(svc,testData,testIndex)
    
    plt.text(0,0, '\n'.join([
    "train error is " + str(percent),
    "test error is " + str(percentTest)
    ]))

    plt.savefig("./data mining/plot/svc/"+ str(name) + ".png", format="png")
    plt.close()

def test(clf,source,index):
    match = 0
    sample = 0
    for item in source:
        Z = clf.predict([item])
        if(Z == index[sample]):
            match+=1
        sample+=1
    return match/sample
    
print(svc.predict([[200000, 40]]))
print(svc.predict([[50000, 65]]))

plotPredictions(svc,'linear')

svc = svm.SVC(kernel='poly', C=C,max_iter=500).fit(data, index)
plotPredictions(svc, 'poly')

svc = svm.SVC(kernel='rbf', C=C,max_iter=500).fit(data, index)
plotPredictions(svc, 'rbf')

svc = svm.SVC(kernel='sigmoid', C=C,max_iter=500).fit(data, index)
plotPredictions(svc, 'sigmoid')
