import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

x = np.arange(-3,3,0.001)

plt.plot(x, norm.pdf(x),'b-.')
plt.plot(x, norm.pdf(x,1,0.5),'r.')
plt.savefig("./matplot/plots/line_types_and_colors.png",format="png")
