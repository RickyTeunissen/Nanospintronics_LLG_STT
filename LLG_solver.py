""""
Within this file, all functions relevant for solving the LLG ODE are present:
- The actual ODE solver itself: LLG_solver()
- The LLG equation together with functions for relevant parameters such as e.g. the effective magnetic field, spin
    torque, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def LLG(t, m3d: tuple, H: tuple, alpha: float = 0.01, Ms: float = 0.127e7):
    """
    returns the right hand side of the IMPLICIT LLG equation

    Maybe later watch out with e.g. units (kA/m vs A/m, normalized units?,...)

    :param t: later used by ODE solve to store time step
    :param m3d: tuple(mx,my,mz) = Unit vector,  holds  the current magnetization
    :param H: tuple(Hx,Hy,Hz): the magnetic field in [A/m]
    :param alpha: float: damping constant
    :param Ms: Saturation magnetization in [A/m]
    :return: (dmx,dmy,dmz)
    """
    gyro_ratio = 8.8e10  # later modify
    mu0 = 1.256e-6  # later modify?

    dmx, dmy, dmz = -gyro_ratio * mu0 / (1 + alpha ** 2) * (np.cross(m3d, H) + alpha * np.cross(m3d, np.cross(m3d, H)))

    return dmx, dmy, dmz


def LLG_solver(IC:tuple, t_points: np.array, H:tuple, alpha: float = 0.01, Ms:float = 0.127e7):
    """
    Solves the LLG equaiton for given IC and parameters in the time range t_points

    :param IC: (mx0,my0,mz0) Initial M direction:
    :param t_points: Array of time points at which we wanna evaluate the solution
    :param H: tuple(Hx,Hy,Hz): the magnetic field in [A/m]
    :param alpha: float: damping constant
    :param Ms: Saturation magnetization in [A/m]
    :return: unit vector m over time in
    """

    tspan = [t_points[0], t_points[-1]]  # ODE solver needs to know t bounds in advance

    parameters = (H, alpha, Ms)
    LLG_sol = solve_ivp(LLG, y0=IC, t_span=tspan, t_eval=t_points, args=parameters)

    mx_sol = LLG_sol.y[0, :]
    my_sol = LLG_sol.y[1, :]
    mz_sol = LLG_sol.y[2, :]
    return mx_sol, my_sol, mz_sol


if __name__ == "__main__":
    # let's run an example of the LLG solver for some IC.
    # NOTE: MUST ALWAYS START AT SOME ANGLE AS ELSE MAYBE ISSUE's when temp = 0

    H = np.array([10e3, 10e3, 10e4])
    m0 = np.sqrt(np.array([1, 1, 0]))
    t = np.arange(0, 10e-9, 1e-11)
    mx, my, mz = LLG_solver(m0, t, H)

    f = plt.figure(1)
    axf = f.add_subplot(projection="3d")
    axf.plot(mx, my, mz, "b-")
    axf.scatter(1, 1, 0, color="red", lw=10)
    axf.set_xlabel("mx")
    axf.set_ylabel("my")
    axf.set_zlabel("mz")
    axf.set_title("Some stupid example")

    axf.set_xlim([-2, 2])
    axf.set_ylim([-2, 2])
    axf.set_zlim([-2, 2])

    plt.show()
