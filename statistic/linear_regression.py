import numpy as np
from pylab import *
import matplotlib.pyplot as plt 
from scipy import stats

pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = 100 - (pageSpeeds +  np.random.normal(0, .1, 1000)) * 3

slope, intercept, r_value, p_value, std_err = stats.linregress(pageSpeeds, purchaseAmount)

plt.scatter(pageSpeeds, purchaseAmount)
plt.text(3.5,95, '\n'.join([
    "slope is " + str(slope),
    "intercept is " + str(intercept),
    "r_value is " + str(r_value), 
    "R_squared is " + str(r_value**2),
    "p_value is " + str(p_value),
    "std_err is " + str(std_err)]
    ))

def predict(x):
    return slope * x + intercept

fitLine = predict(pageSpeeds)

plt.scatter(pageSpeeds, purchaseAmount)
plt.plot(pageSpeeds, fitLine, c='r')

plt.show()