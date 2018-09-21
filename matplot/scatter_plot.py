import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from pylab import randn

X = randn(500)
Y = randn(500)
plt.scatter(X,Y)
plt.savefig("./matplot/plots/scatter_plot.png",format="png")