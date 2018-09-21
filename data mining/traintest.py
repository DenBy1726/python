import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)

#data set
pageSpeeds = np.random.normal(3.0, 1.0, 100)
purchaseAmount = np.random.normal(50.0, 30.0, 100) / pageSpeeds


plt.scatter(pageSpeeds, purchaseAmount)
plt.savefig("./data mining/plot/traintest_data.png", format="png")
plt.close()

#train set 80%, test set 20%
trainX = pageSpeeds[:80]
testX = pageSpeeds[80:]

trainY = purchaseAmount[:80]
testY = purchaseAmount[80:]

x = np.array(trainX)
y = np.array(trainY)

#approximate train set
p4 = np.poly1d(np.polyfit(x, y, 8))
xp = np.linspace(0, 7, 100)

axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])

#check regression for train set
plt.scatter(x, y)
plt.plot(xp, p4(xp), c='r')
r2 = r2_score(trainY, p4(trainX))
plt.legend(["r2 is " + str(r2)])
plt.savefig("./data mining/plot/traintest_train.png", format="png")
plt.close()

testx = np.array(testX)
testy = np.array(testY)

axes = plt.axes()
axes.set_xlim([0,7])
axes.set_ylim([0, 200])

#check regression for test set
plt.scatter(testx, testy)
plt.plot(xp, p4(xp), c='r')
r2 = r2_score(testy, p4(testx))
plt.legend(["r2 is " + str(r2)])
plt.savefig("./data mining/plot/traintest_test.png", format="png")
plt.close()