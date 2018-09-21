import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

incomes = np.floor(np.random.normal(27000,15000,10001))
print(np.mean(incomes))

plt.hist(incomes,50)

#plt.show()

print(np.median(incomes))

print()

incomes = np.append(incomes, [100000000])
print(np.mean(incomes))
print(np.median(incomes))

print()

ages = np.random.randint(18, 90, 500)
mode = stats.mode(ages)
print(mode)