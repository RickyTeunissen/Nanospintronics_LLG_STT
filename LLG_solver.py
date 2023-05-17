"""
Within this file, all functions relevant for solving the LLG ODE are present:
- The actual ODE solver itself: LLG_solver()
- The LLG equation together with functions for relevant parameters such as e.g. the effective magnetic field, spin
    torque, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.constants as constants
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator

mu0: float = constants.mu_0  # [N/A^2]
hbar: float = constants.hbar
me: float = constants.electron_mass  # [kg]
charge: float = constants.elementary_charge  # [C]
gyro_ratio: float = 2 * charge / (2 * me)  # [A/m]


# gyro_ratio: float = 1.760896 * 10 ** 11  # [A/m]


def calc_effective_field(m3d: tuple, Hext: np.array, Ms: float) -> tuple:
    """
    Calculates the effective field based on a given state of the system.
    Takes into account:
    - demag field
    - anisotropic field (NOT YET)

    :param m3d: 3d tuple for the unit vector of the magnetizaiton (mx,my,mz)
    :param Ms: Saturation magnetizaion in [A/m]
    :param Hext: The externally applied field in [A/m]
    :return: Heff = (Heffx, Heffy, Heffz) = effective field in [A/m]
    """
    # Still add magnetocrystalline ani?:

    # we use thin films, these have a demagnetization field:
    H_demag = (0, 0, -m3d[2] * Ms)

    Heff = H_demag + Hext

    return Heff


def calc_total_spin_torque():
    pass


def N(J: float, d: float, m3d: np.array, M3d: np.array) -> np.array:
    eta = 1
    N_spin_transfer = eta * (hbar / (2 * charge)) * (J / d) * np.cross(m3d, np.cross(m3d, M3d))
    return N_spin_transfer


def LLG(t, m3d: tuple, Hext: np.array, alpha: float, Ms: float, J: float, d: float, M3d: tuple):
    """
    returns the right hand side of the IMPLICIT LLG equation

    Maybe later watch out with e.g. units (kA/m vs A/m, normalized units?,...)

    :param t: later used by ODE solve to store time step
    :param m3d: tuple(mx,my,mz) = Unit vector, magnetization of the free layer
    :param Hext: tuple(Hx,Hy,Hz): the magnetic field in [A/m]
    :param alpha: damping constant
    :param Ms: Saturation magnetization in [A/m]
    :param J: The current area density through the spin valve [A/m^2]
    :param d: The thickness of the free layer [m]
    :param M3d: tuple(Mx,My,Mz) = Unit vector, the magnetization of the fixed layer
    :return: (dmx,dmy,dmz)
    """

    Heff = calc_effective_field(m3d, Hext, Ms)
    basics = (-gyro_ratio * mu0 / (1 + alpha ** 2)) * (np.cross(m3d, Heff) + alpha * np.cross(m3d, np.cross(m3d, Heff)))
    spinTorque = (gyro_ratio / (mu0 * Ms)) * N(J, d, m3d, M3d)

    dmx, dmy, dmz = basics + spinTorque

    return dmx, dmy, dmz


def LLG_solver(IC: tuple, t_points: np.array, Hext: np.array, alpha: float, Ms: float, J: float, d: float,
               M3d: tuple) -> tuple:
    """
    Solves the LLG equation for given IC and parameters in the time range t_points

    :param IC: (mx0,my0,mz0) Initial magnetization of the free layer:
    :param t_points: Array of time points at which we wanna evaluate the solution
    :param Hext: tuple(Hx,Hy,Hz): the external magnetic field in [A/m]
    :param alpha: damping constant
    :param Ms: Saturation magnetization in [A/m]
    :param J: The current area density through the spin valve [A/m^2]
    :param d: The thickness of the free layer [m]
    :param M3d: tuple(Mx,My,Mz) = Unit vector, the magnetization of the fixed layer
    :return: unit vector m over time in
    """

    tspan = [t_points[0], t_points[-1]]  # ODE solver needs to know t bounds in advance

    parameters = (Hext, alpha, Ms, J, d, M3d)
    LLG_sol = solve_ivp(LLG, y0=IC, t_span=tspan, t_eval=t_points, args=parameters)

    mx_sol = LLG_sol.y[0, :]
    my_sol = LLG_sol.y[1, :]
    mz_sol = LLG_sol.y[2, :]
    return mx_sol, my_sol, mz_sol


def polarToCartesian(r: float, theta: float, phi: float) -> np.array:
    """
    :param r: distance to origin [m]
    :param theta: angle with z axis in [radians]
    :param phi: angle with x axis in [radians]
    :return:
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def plotResult(mx: np.array, my: np.array, mz: np.array, m0: np.array, t: np.array):
    f = plt.figure(1)
    axf = f.add_subplot(projection="3d")
    axf.plot(mx, my, mz, "b-")
    axf.scatter(m0[0], m0[1], m0[2], color="red", lw=5)
    axf.set_xlabel("mx")
    axf.set_ylabel("my")
    axf.set_zlabel("mz")
    axf.set_title("Magnetization direction over time")
    axf.xaxis.set_major_locator(MultipleLocator(0.5))
    axf.yaxis.set_major_locator(MultipleLocator(0.5))
    axf.zaxis.set_major_locator(MultipleLocator(0.5))
    axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.set_xlim([-1, 1])
    axf.set_ylim([-1, 1])
    axf.set_zlim([-1, 1])

    fig2, ax = plt.subplots(3, 1)
    ax[0].plot(t, mx)
    ax[0].set_ylabel("mx")
    ax[0].set_ylim([-1.1, 1.1])
    ax[1].plot(t, my)
    ax[1].set_ylabel("my")
    ax[1].set_ylim([-1.1, 1.1])
    ax[2].plot(t, mz)
    ax[2].set_ylabel("mz")
    ax[2].set_ylim([-1.1, 1.1])
    plt.show()


if __name__ == "__main__":
    # let's run an example of the LLG solver for some IC.
    # NOTE: MUST ALWAYS START AT SOME ANGLE AS ELSE MAYBE ISSUE's when temperature = 0

    Hext = np.array([-3.9e4, 0, 0])  # [A/m]
    m0 = polarToCartesian(1, 0.49*np.pi, 0.1)
    alpha = 0.1
    Ms = 1.27e6  # [A/m]
    J = 1e8  # [A/m^2]
    d = 3e-9  # [m]
    M3d = np.array([0, 0, 1]) #polarToCartesian(1, 0.5*np.pi, 0)
    t = np.arange(0, 0.5e-10, 5e-16)
    mx, my, mz = LLG_solver(m0, t, Hext, alpha, Ms, J, d, M3d)

    plotResult(mx, my, mz, m0, t)
