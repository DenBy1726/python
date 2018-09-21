import numpy as np
import matplotlib.pyplot as plt

incomes = np.random.normal(100,20,10000)

plt.hist(incomes,50)
plt.show()

print(incomes.std())
print(incomes.var())