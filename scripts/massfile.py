import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../output/sphgauss1/fill/history.csv")

plt.scatter(df["param"] - 0.3360352955138296, df["mass"])
plt.xscale("log")
plt.yscale("log")
plt.show()