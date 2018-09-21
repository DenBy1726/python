import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

vals = np.random.normal(0, 0.5, 10000)

plt.hist(vals, 50)

med1 = np.percentile(vals, 50)
med2 = np.median(vals)

print(np.percentile(vals, 90))
print(np.percentile(vals, 20))

print(med1 == med2)

min1 = np.percentile(vals, 0)
min2 = min(vals)

print(min1 == min2)

max1 = np.percentile(vals, 100)
max2 = max(vals)

print(max1 == max2)

#first moment
print(np.mean(vals))
#second moment
print(np.var(vals))
#third moment
print(st.skew(vals))
#fourth moment
print(st.kurtosis(vals))