import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

uniformSkewed = np.random.rand(100) * 100 - 40
high_outliers = np.random.rand(10) * 50 + 100
low_outliers = np.random.rand(10) * -50 - 100
data = np.concatenate((uniformSkewed, high_outliers, low_outliers))
plt.boxplot(data)

plt.savefig("./matplot/plots/boxplot.png",format="png")