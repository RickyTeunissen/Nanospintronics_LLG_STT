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
import scipy

mu0: float = constants.mu_0  # [N/A^2]
gyro_ratio: float = 1.760896 * 10 ** 8  # [A/m]
hbar: float = constants.hbar
me: float = constants.electron_mass
gyro_ratio1: float = constants.elementary_charge/(2*me)  # [A/m]


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
    N_spin_transfer = np.cross(eta*(hbar/(2*me))*(J/d)*m3d, np.cross(m3d, M3d))
    return N_spin_transfer


def LLG(t, m3d: tuple, Hext: np.array, alpha: float, Ms: float):
    """
    returns the right hand side of the IMPLICIT LLG equation

    Maybe later watch out with e.g. units (kA/m vs A/m, normalized units?,...)

    :param t: later used by ODE solve to store time step
    :param m3d: tuple(mx,my,mz) = Unit vector,  holds  the current magnetization
    :param Hext: tuple(Hx,Hy,Hz): the magnetic field in [A/m]
    :param alpha: float: damping constant
    :param Ms: Saturation magnetization in [A/m]
    :return: (dmx,dmy,dmz)
    """

    Heff = calc_effective_field(m3d, Hext, Ms)
    # spinTorque = (gyro_ratio / (mu0 * Ms)) * N

    spin_precession = -gyro_ratio * mu0 / (1 + alpha**2) * (np.cross(m3d, Heff))
    gilbert_damping = alpha * np.cross(m3d, np.cross(m3d, Heff))
    dmx, dmy, dmz = spin_precession + gilbert_damping

    return dmx, dmy, dmz


def LLG_solver(IC: tuple, t_points: np.array, Hext: np.array, alpha: float = 0.05, Ms: float = 1.27e6):
    """
    Solves the LLG equation for given IC and parameters in the time range t_points

    :param IC: (mx0,my0,mz0) Initial M direction:
    :param t_points: Array of time points at which we wanna evaluate the solution
    :param Hext: tuple(Hx,Hy,Hz): the external magnetic field in [A/m]
    :param alpha: float: damping constant
    :param Ms: Saturation magnetization in [A/m]
    :return: unit vector m over time in
    """

    tspan = [t_points[0], t_points[-1]]  # ODE solver needs to know t bounds in advance

    parameters = (Hext, alpha, Ms)
    LLG_sol = solve_ivp(LLG, y0=IC, t_span=tspan, t_eval=t_points, args=parameters)

    mx_sol = LLG_sol.y[0, :]
    my_sol = LLG_sol.y[1, :]
    mz_sol = LLG_sol.y[2, :]
    return mx_sol, my_sol, mz_sol


def polarToCartesian(r: float, theta: float, phi: float) -> np.array:
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


if __name__ == "__main__":
    # let's run an example of the LLG solver for some IC.
    # NOTE: MUST ALWAYS START AT SOME ANGLE AS ELSE MAYBE ISSUE's when temperature = 0

    Hext = np.array([0, 0.2e6, 0])  # external field in the y direction
    m0 = polarToCartesian(1, 0.2, 0)
    t = np.arange(0, 1e-9, 0.1e-12)
    mx, my, mz = LLG_solver(m0, t, Hext)

    f = plt.figure(1)
    axf = f.add_subplot(projection="3d")
    axf.plot(mx, my, mz, "b-")
    axf.scatter(m0[0], m0[1], m0[2], color="red", lw=10)
    axf.set_xlabel("mx")
    axf.set_ylabel("my")
    axf.set_zlabel("mz")
    axf.set_title("Some stupid example")

    axf.set_xlim([-1.5, 1.5])
    axf.set_ylim([-1.5, 1.5])
    axf.set_zlim([-1.5, 1.5])

    fig2, ax = plt.subplots(3, 1)
    ax[0].plot(t, mx)
    ax[1].plot(t, my)
    ax[2].plot(t, mz)
    plt.show()
