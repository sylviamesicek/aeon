import numpy as np

# ratios = open("output/ratios.csv")
# ratios_s = ratios.read()
# ratios_n = ratios_s.replace(",", " ")
# ratios.close()

# n = open("new.csv", "w")
# n.write(ratios_n)
# n.close()

data = np.loadtxt("new.csv")
datat = np.transpose(data)

ratios = datat[0]
amplitudes = datat[1]

v = np.argsort(amplitudes)

ratios_sorted = ratios[v]
amplitudes_sorted = amplitudes[v]

print(ratios_sorted, amplitudes_sorted)

import matplotlib.pyplot as plt

x = amplitudes_sorted
y = np.log2(1/ratios)

# plt.title("Convergance Rates of FDM Solver")
plt.xlabel("Scalar Field Amplitude")
plt.ylabel("Order of Convergence")
plt.xlim(1.0, 9.0)
plt.ylim(3, 5)
plt.plot(x, y)
plt.show()

plt.savefig("output/convergence.eps")