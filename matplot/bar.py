import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

values = [11,42,21,12,14]
colors = ['r','g','b','c','m']

plt.bar(range(0,5),values,color=colors)
plt.savefig("./matplot/plots/bar.png",format="png")