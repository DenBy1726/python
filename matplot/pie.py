import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


x = np.arange(-3,3,0.001)

values = [15,25,23,18,19]
explode = [0,0,0.1,0,0]
labels = ['Fifteen', 'Twenty five', 'Twenty three', 'Eighteen', 'Nineteen']

plt.title("Title")
plt.pie(values,explode,labels)
plt.savefig("./matplot/plots/pie.png",format="png")