import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.arange(-3,3,0.001)

plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x,1,0.5))

plt.xlabel('X Axes')
plt.ylabel('Y Axes')
plt.legend(["Default normal", "Custom normal"], loc=4)

plt.savefig("./matplot/plots/labels.png",format="png")