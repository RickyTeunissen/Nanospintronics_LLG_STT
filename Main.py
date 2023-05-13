import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

#just some test stuff, can later all be removed
print("Hello world")
print("Programmed to work and not to feeeeel!")

test = np.linspace(1, 20)
print(test)


fig,ax = plt.subplots(2,2, figsize = (10,5))
ax[0, 0].plot(test, test**2)
ax[0, 0].set_xlabel("lol")
ax[0, 0].set_ylabel("hihi")
plt.show()

