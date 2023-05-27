"""
This file is mainly used for testing/optimizing time efficiency and for playing around with stuff.
If one wants a specific functionalaty, please use one of the files belonging to it.
"""

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import scipy
# import time
import cProfile
from faster_LLG_solver import *


# # just some test stuff, can later all be removed
# print("Hello world")
# print("Programmed to work and not to feeeeel!")
#
# test = np.linspace(1, 20)
# print(test)
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 5))
# ax[0, 0].plot(test, test ** 2)
# ax[0, 0].set_xlabel("lol")
# ax[0, 0].set_ylabel("hihie")
#
# n = 50
# x = np.linspace(0, 5, n)  # in our code i = np.arange(Imin,Imax,stepsize)
# y = np.linspace(0, 5, n)  # in our code h = np.arange(Hmin,Hmax,stepsize)
# X, Y = np.meshgrid(x, y)  # Yes we also need to turn into meshgrid
# z = np.sin(X * Y ** 2) - np.cos(X)
# # in our code we need some array storing all the results in the form of [[row 1 (all H=H1)],[row 2 (all H=H2) ],[]]
# print(z.shape,X.shape,x.shape)
#
# fig2 = plt.figure(figsize=(6, 6))
# plt.pcolormesh(X, Y, z, cmap=cm.Blues, )
# plt.colorbar()
#
# plt.show()

def example_runner():
    Hext = np.array([-5.5e4, 0, 0])  # [A/m]
    alpha = 0.01  # SHOULD BE 0.01 FOR Cu!
    Ms = 1.27e6  # [A/m]
    K_surface = 0.5e-3  # J/m^2
    J = 0.2e12  # [A/m^2]
    d = 3e-9  # [m]
    width_x = 130e-9  # [m] need width_x > width_y >> thickness (we assume super flat ellipsoide)
    width_y = 70e-9  # [m]
    temperature = 3  # [K], note: need like 1e5 to see really in plot (just like MATLAB result)

    # initial direction free layer and fixed layer
    m0 = np.array([1, 0, 0])
    M3d = np.array([1, 0, 0])

    # which t points solve for, KEEP AS ARANGE (need same distance between points)!!
    t = np.arange(0, 100e-9, 1e-12)

    # solving the system
    mx, my, mz = LLG_solver(m0, t, Hext, alpha, Ms, J, d, width_x, width_y, temperature, M3d, K_surface)


# plotResult(mx, my, mz, m0, t)

if __name__ == "__main__":
    print("Timing programm...")
    cProfile.run("example_runner()", sort="tottime")
