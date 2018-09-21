from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

plt.scatter(pageSpeeds, purchaseAmount)

x = np.array(pageSpeeds)
y = np.array(purchaseAmount)

p4 = np.poly1d(np.polyfit(x, y, 4))

xp = np.linspace(0, 7, 100)
plt.plot(xp, p4(xp), c='r')
r2 = r2_score(y, p4(x))

plt.text(3,1,"regression is " + str(r2))
plt.savefig("./statistic/plots/regression4.png",format="png")

plt.close()

regs = []
for deg in range(0,15):
    p4 = np.poly1d(np.polyfit(x, y, deg))
    xp = np.linspace(0, 7, 100)
    r2 = r2_score(y, p4(x))
    regs.append(r2)

plt.plot(range(0,15), regs, c='r')
plt.xlabel("degree")
plt.ylabel("regression")
axes = plt.axes()
axes.set_xlim([0,15])
axes.set_ylim([0,1])
plt.title("Plot of depending degree polynom on regression percent")
plt.savefig("./statistic/plots/regressiondegree.png",format="png")
