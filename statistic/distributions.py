import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import binom
from scipy.stats import poisson

values = np.random.uniform(-10.0, 10.0,100000)
#plt.hist(values, 50)

x = np.arange(-3,10,0.001)
binom_x = np.arange(0,10,0.001)
poisson_x = np.arange(400,600,0.5)

mu = 500

norm_pdf = norm.pdf(x)
exp_pdf = expon.pdf(x[3001:])
binom_pmf = binom.pmf(binom_x,10,0.5)
poisson_pmf = poisson.pmf(poisson_x,mu)

plt.plot(x,norm_pdf)
plt.plot(x[3001:],exp_pdf)
plt.plot(binom_x,binom_pmf)
plt.show()

plt.close()

plt.plot(poisson_x,poisson_pmf)
plt.show()

poisson_list = poisson_x.tolist()
given_index = poisson_list.index(550)
prob = sum(poisson_pmf.tolist()[given_index:])

print(prob)