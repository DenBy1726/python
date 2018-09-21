import numpy as np
from pylab import dot,mean
import matplotlib.pyplot as plt
from scipy.stats import norm
from pylab import randn

def de_mean(x):
    xmean = mean(x)
    return [xi - xmean for xi in x]

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n-1)

def correlation(x, y):
    stddevx = x.std()
    stddevy = y.std()
    return covariance(x,y) / stddevx / stddevy

#независимы
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000)

plt.scatter(pageSpeeds, purchaseAmount)
plt.legend(["Covariance is " + str(correlation(pageSpeeds, purchaseAmount))])

plt.savefig("./statistic/plots/cov1.png",format="png")
print(np.corrcoef(pageSpeeds, purchaseAmount))

plt.close()

#теперь переменные зависимы
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
plt.scatter(pageSpeeds, purchaseAmount)
plt.legend(["Covariance is " + str(correlation(pageSpeeds, purchaseAmount))])

plt.savefig("./statistic/plots/cov2.png",format="png")

print(np.corrcoef(pageSpeeds, purchaseAmount))

plt.close()

#линейная зависимость
purchaseAmount = 100 - pageSpeeds * 3
plt.scatter(pageSpeeds, purchaseAmount)
plt.legend(["Covariance is " + str(correlation(pageSpeeds, purchaseAmount))])

plt.savefig("./statistic/plots/cov3.png",format="png")

print(np.corrcoef(pageSpeeds, purchaseAmount))

print(covariance (pageSpeeds, purchaseAmount))
print(np.cov(pageSpeeds, purchaseAmount))
