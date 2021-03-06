import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


x = np.arange(-3,3,0.001)

plt.plot(x, norm.pdf(x))
plt.show()