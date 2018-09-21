import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.arange(-3,3,0.001)

axes = plt.axes()
axes.set_xlim([-5,5])
axes.set_ylim([0,1])
axes.set_xticks(np.arange(-5,5,1))
axes.set_yticks(np.arange(0,1,0.1))
axes.grid()
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x,1,0.5))
plt.savefig("./matplot/plots/adjust_axes.png",format="png")
