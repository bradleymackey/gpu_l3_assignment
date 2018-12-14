import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random
import queue



xdata = [3580,8950,17900,31325,125091,125091]
ydata = [0.054,0.358,2.321,11.09,7.449,2.082]

#plot degree distribution 
plt.xlabel("Matrix Size")
plt.ylabel("Total Runtime")
plt.title("Runtime vs Matrix Size (summation) (large)")
plt.plot(xdata, ydata, marker=".", linestyle="None", color="b")
plt.savefig("large-sum-n.png")