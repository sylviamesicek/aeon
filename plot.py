import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("output/strong1.4.txt")
data = np.transpose(data)

plt.scatter(data[0], data[1])
plt.xlabel("Proper Time")
plt.ylabel("Pi^2 at origin")
plt.show()