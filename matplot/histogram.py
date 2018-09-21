import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


x = np.random.normal(27000,15000,15000)
plt.hist(x,50)

plt.savefig("./matplot/plots/histogram.png",format="png")