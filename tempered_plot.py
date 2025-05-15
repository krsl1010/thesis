import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import norm 

def bimodal_gaussian(x, T=1):
    return 0.5 * norm.pdf(x, -10, 1)**(1/T) + 0.5 * norm.pdf(x, 10, 1)**(1/T)

x = np.linspace(-15, 15, 1000)
temps = [1, 10, 0.1]
plt.figure()
plt.suptitle("Bimodal Gaussian distribution")
plt.subplot(3, 1, 1)
plt.plot(x, bimodal_gaussian(x))
plt.xlabel("x")
plt.ylabel("Density")
#plt.title(f"Temperature: {1}")
plt.subplot(3, 1, 2)
plt.plot(x, bimodal_gaussian(x, T=10))
plt.xlabel("x")
plt.ylabel("Density")
#plt.title(f"Temperature: {10}")
plt.subplot(3, 1, 3)
plt.plot(x, bimodal_gaussian(x, T=0.1))
plt.xlabel("x")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("bimodal_gaussian.pdf")
plt.show()
