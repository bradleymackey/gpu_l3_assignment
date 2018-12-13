import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random
import queue



xdata = [1750,3500,5800,8700,1860,1860]
ydata = [2.7,31,150,520,3.6,3.7]

#plot degree distribution 
plt.xlabel("Matrix Size")
plt.ylabel("Total Runtime")
plt.title("Runtime vs Matrix Size (small)")
plt.plot(xdata, ydata, marker=".", linestyle="None", color="b")
plt.savefig("large-n.png")