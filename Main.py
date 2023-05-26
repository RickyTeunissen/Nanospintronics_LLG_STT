import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import time

# just some test stuff, can later all be removed
print("Hello world")
print("Programmed to work and not to feeeeel!")

test = np.linspace(1, 20)
print(test)

fig, ax = plt.subplots(2, 2, figsize=(10, 5))
ax[0, 0].plot(test, test ** 2)
ax[0, 0].set_xlabel("lol")
ax[0, 0].set_ylabel("hihie")

n = 50
x = np.linspace(0, 5, n)  # in our code i = np.arange(Imin,Imax,stepsize)
y = np.linspace(0, 5, n)  # in our code h = np.arange(Hmin,Hmax,stepsize)
X, Y = np.meshgrid(x, y)  # Yes we also need to turn into meshgrid
z = np.sin(X * Y ** 2) - np.cos(X)
# in our code we need some array storing all the results in the form of [[row 1 (all H=H1)],[row 2 (all H=H2) ],[]]
print(z.shape,X.shape,x.shape)

fig2 = plt.figure(figsize=(6, 6))
plt.pcolormesh(X, Y, z, cmap=cm.Blues, )
plt.colorbar()

plt.show()
