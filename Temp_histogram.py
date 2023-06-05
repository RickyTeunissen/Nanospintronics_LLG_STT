"""
Using this file we can just quickly run th LLG solver in presence of only a field and some temperature and look
at the histogram of Mz values encountered to ensure that the right thermal distrubution is obtained using our code

NOTE:
To make this program work, we need to turn of all anisotropies, also e.g. shape anisotropy, the easiest way to do this
is to go to the LLG solver code and manually set the demag tensor to [0,0,0]
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from LLG_solver import *
from scipy.optimize import curve_fit


def fit_func(x, Temp, A):
    kb = scipy.constants.Boltzmann
    return A*np.exp(-x/(kb*Temp))

if __name__ == "__main__":
    # let's run an example of the LLG solver for some IC.
    # NOTE: MUST ALWAYS START AT SOME ANGLE WHEN TEMPERATURE=0
    print("Starting trajectory calculation....")
    start = time.time()

    # defining relevant system parameters:
    # need to have no shape anisotropy and no STT and no surface ani and.... basically only some Hext and T
    Hext = np.array([3.9808e4, 0, 0])  # 50mT in [A/m]
    alpha = 0.1
    Ms = 1.27e6  # [A/m]
    K_surface = 0  # 0.5e-3  # J/m^2
    J = 0  # [A/m^2]

    # these values are arbitrary and can be set to anything, BUT SET DEMAG TENSOR TO 0,0,0 IN LLG SOLVER!!!
    d = 3e-9  # [m]
    width_x = 130e-9  # [m] need width_x > width_y >> thickness (we assume super flat ellipsoide)
    width_y = 70e-9  # [m]

    temperature = 3000  # [K], note: need like 1e5 to see really in plot (just like MATLAB result)

    # initial direction free layer and fixed layer
    m0 = np.array([1, 0, 0])
    M3d = np.array([1, 0, 0])

    # which t points solve for, KEEP AS ARANGE (need same distance between points)!!
    # need long time s.t. really get thermal distribution
    t = np.arange(0, 1e-8, 1e-12) #np.arange(0, 5e-7, 1e-12)

    # solving the system
    mx, my, mz = LLG_solver(m0, t, Hext, alpha, Ms, J, d, width_x, width_y, temperature, M3d, K_surface)

    # converting the m values to zeeman energies in kbt (choose reference s.t. en if Parralel =0):
    volume = np.pi * width_x / 2 * width_y / 2 * d
    #volume = 100e-9 ** 3
    en_array = volume*Ms*mu0*Hext[0]*(1-mx)
    print(np.mean(en_array), kb*temperature)

    # calculate prob density for what we found and a fit to see if Temp correct.
    # assume first few steps needed to go to thermal equilibrium
    counts, bins = np.histogram(en_array[10000:], bins=50)
    prob = counts / sum(counts)
    centers = (bins[:-1] + bins[1:]) / 2
    popt, pcov = curve_fit(fit_func, centers, prob, p0=[temperature,0.2])
    print(popt)
    print(pcov)
    print(f"Fitting found T={popt[0]} and A = {popt[1]}")

    # plot theoretical and measured
    figf = plt.figure(5)
    plt.plot(centers, fit_func(centers,*popt))
    plt.stairs(prob, bins)
    plt.xlabel("Energie [J]")
    plt.ylabel("P(E)")
    plt.legend(["Fit $A e^{-E/K_B T}$"+f", T={popt[0]:.1f}K", "Measurements"])
    plt.show()

    plotResult(mx,my,mz,m0,t)

