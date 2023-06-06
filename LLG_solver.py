"""
Within this file, all functions relevant for solving the LLG ODE are present:
- The actual ODE solver itself: LLG_solver()
- The LLG equation together with functions for relevant parameters such as e.g. the effective magnetic field, spin
    torque, etc.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.constants as constants
import scipy.special
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator
from math import sqrt, floor, sin

# Defining relevant physical constants.
mu0: float = constants.mu_0  # [N/A^2]
hbar: float = constants.hbar
me: float = constants.electron_mass  # [kg]
charge: float = constants.elementary_charge  # [C]
gyro_ratio: float = 2 * charge / (2 * me)  # [A/m]
kb: float = constants.Boltzmann  # [m^2 kg S^-2 K^-1]

gyro_co = 2.4e5  # Gyromagnetic ratio for Co, copied from DOI: 10.1103/PhysRevB.72.014446 in [m/As]


def better_cross(a, b):
    """
    calculates the cross product, but without the overhead that numpy has

    NOTE: both vectors MUST be 3d else this function will be wrong!
    :param a: np.array 3x1
    :param b: np.array 3x1
    :return: c = a cross b

    """
    cx = a[1] * b[2] - a[2] * b[1]
    cy = a[2] * b[0] - a[0] * b[2]
    cz = a[0] * b[1] - a[1] * b[0]
    return np.array([cx, cy, cz])


def calc_effective_field(m3d: tuple, Hext: np.array, Ms: float, H_temp_array: np.array, current_time: float,
                         t_stepsize: float, demag_tens: np.array, K_surf: float, thickness: float) -> tuple:
    """
    Calculates the effective field based on a given state of the system.
    Takes into account:
    - demag field
    - surface anisotropy
    - stochatsic field due to thermal fluctuations

    :param m3d: 3d tuple for the unit vector of the magnetizaiton (mx,my,mz)
    :param Hext: The externally applied field in [A/m]
    :param Ms: Saturation magnetizaion in [A/m]
    :param H_temp_array: array of random fields over time due to temperature (number timesteps)x3
    :param current_time: current time where the ODE solver is solving
    :param t_stepsize: stepsize in [s] between ODE solution points
    :param demag_tens: array of all diagonal components of the demag tensor
    :param K_surf: surface anisotropy that wants sys to go OOP [J/m]
    :param thickness: thickness cylinder in [m]
    :return: Heff = (Heffx, Heffy, Heffz) = effective field in [A/m]
    """

    # Maybe add T dependent anisotropy

    # we use thin films, these have a demagnetization field:
    H_demag = -Ms * demag_tens * m3d

    # additional contribution due to temperature fluctuations first calculate in which step we are (s.t. in same t step
    # always pick same value if call this function > once)
    step_index = floor(current_time / t_stepsize)
    H_temp = H_temp_array[step_index]

    # contribution by surface anistropy
    # H_surf_ani = np.array([0, 0, 4 * K_surf / (mu0 * Ms * thickness) * (1 - m3d[2] ** 2)])    WRONG
    H_surf_ani = np.array([0, 0, 4 * K_surf / (mu0 * Ms * thickness) * m3d[2]])


    Heff = Hext + H_demag + H_temp + H_surf_ani

    return Heff

def LLG(t, m3d: np.array, Hext: np.array, alpha: float, Ms: float, J: float, d: float, M3d: tuple,
        H_temp_array: np.array, t_stepsize, demag_tensor: np.array, K_surf: float, Jformula):
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
    :param H_temp_array: array of random fields over time due to temperature (number timesteps)x3
    :param demag_tensor: array of all diagonal components of the demag tensor
    :param K_surf: surface anisotropy that wants sys to go OOP [J/m]
    :param Jformula: The lambda expression for the value of J.
    :return: (dmx,dmy,dmz)
    """
    m3d -= m3d - m3d / sqrt(sum(m3d ** 2))

    Jtot = J * Jformula(t)
    # calculate preceission + damping:
    Heff = calc_effective_field(m3d, Hext, Ms, H_temp_array, t, t_stepsize, demag_tensor, K_surf, d)
    preces_damp = (-gyro_ratio * mu0 / (1 + alpha ** 2)) * (
            better_cross(m3d, Heff) + alpha * better_cross(m3d, better_cross(m3d, Heff)))

    # calculate contribution normal spin transfer torque (look my notes if wanna add eta)
    pre = -Jtot / charge * gyro_ratio * hbar / (2 * Ms * d) * 1 / (1 + alpha ** 2)
    spinTorque = pre * (better_cross(m3d, better_cross(m3d, M3d)) - alpha * better_cross(m3d, M3d))

    dmx, dmy, dmz = preces_damp + spinTorque

    return dmx, dmy, dmz


def LLG_solver(IC: np.array, t_points: np.array, Hext: np.array, alpha: float, Ms: float, J: float, thickness: float,
               width_x: float, width_y: float, temp: float, M3d: np.array, K_surf: float, Jformula) -> tuple:
    """
    Solves the LLG equation for given IC and parameters in the time range t_points

    :param IC: (mx0,my0,mz0) Initial magnetization of the free layer:
    :param t_points: Array of time points at which we wanna evaluate the solution
    :param Hext: tuple(Hx,Hy,Hz): the external magnetic field in [A/m]
    :param alpha: damping constant
    :param Ms: Saturation magnetization in [A/m]
    :param J: The current area density through the spin valve [A/m^2]
    :param thickness: The thickness of the free layer [m]
    :param width_x: total width cylinder in longest direction [m]
    :param width_y: total width cylinder in smallest direction [m]
    :param temp: temperature of system [K]
    :param M3d: tuple(Mx,My,Mz) = Unit vector, the magnetization of the fixed layer
    :param K_surf: surface anisotropy that wants sys to go OOP [J/m]
    :param Jformula: The lambda expression for the value of J.

    :return: unit vector m over time in
    """

    tspan = [t_points[0], t_points[-1]]  # ODE solver needs to know t bounds in advance
    t_step_size = t_points[1] - t_points[0]

    # calculate demag tensors for system, based on https://doi.org/10.1103/PhysRev.67.351 eq(2.23)-(2.25):
    a, b, c = width_x / 2, width_y / 2, thickness / 2
    argument = sqrt(1 - (b / a) ** 2)
    F_ellip_int = scipy.special.ellipk(argument)
    E_ellip_int = scipy.special.ellipe(argument)
    Nx = c / a * sqrt(1 - argument ** 2) * (F_ellip_int - E_ellip_int) / argument ** 2
    Ny = c / a * (E_ellip_int - (1 - argument ** 2) * F_ellip_int) / (argument ** 2 * sqrt(1 - argument ** 2))
    Nz = 1 - c / a * E_ellip_int / sqrt(1 - argument ** 2)
    demag_tensor = np.array([Nx, Ny, Nz])  # to test: print(demag_tensor, np.sum(demag_tensor))

    # precompute random field array due to temperature:
    vol = np.pi * a * b * thickness
    # H_temp_std = sqrt(2 * kb * alpha * temp / (mu0 * Ms ** 2 * vol) / t_step_size) WRONG FORM
    H_temp_std = sqrt(2 * kb * alpha * temp / (mu0 * Ms * gyro_co * vol * t_step_size))
    H_temp_arr = np.random.normal(0, H_temp_std, [t_points.shape[0], 3])  # gives array(meas points, 3) for all H terms

    # solve the ODE
    parameters = (Hext, alpha, Ms, J, thickness, M3d, H_temp_arr, t_step_size, demag_tensor, K_surf, Jformula)
    LLG_sol = solve_ivp(LLG, y0=IC, t_span=tspan, t_eval=t_points, args=parameters, method="RK45")

    # pack out solutions
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
    f = plt.figure(1, figsize=(8, 7))
    axf = f.add_subplot(projection="3d")
    axf.plot(mx, my, mz, "b-", lw=0.3)
    last_part = np.floor(mz.shape[0] * 0.1).astype(int)  # draw last 10% green to show e.g. stable orbit shape
    axf.plot(mx[-last_part:], my[-last_part:], mz[-last_part:], color="lime", lw=2)
    axf.scatter(m0[0], m0[1], m0[2], color="red", lw=3)  # startpoint
    axf.set_xlabel("mx/Ms")
    axf.set_ylabel("my/Ms")
    axf.set_zlabel("mz/Ms")
    axf.set_title("Magnetization direction over time", fontweight="bold")
    axf.xaxis.set_major_locator(MultipleLocator(0.5))
    axf.yaxis.set_major_locator(MultipleLocator(0.5))
    axf.zaxis.set_major_locator(MultipleLocator(0.5))
    axf.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axf.set_xlim([-1, 1])
    axf.set_ylim([-1, 1])
    axf.set_zlim([-1, 1])

    fig2, ax = plt.subplots(3, 1, sharex=True)
    fig2.suptitle("Magnetization over time", fontweight="bold")
    ax[0].plot(t, mx, lw=0.6)
    ax[0].set_ylabel("mx/Ms")
    ax[0].set_ylim([-1.1, 1.1])
    ax[0].tick_params(direction="in", bottom=True, top=True, left=True, right=True)
    ax[1].plot(t, my, lw=0.6)
    ax[1].set_ylabel("my/Ms")
    ax[1].set_ylim([-1.1, 1.1])
    ax[1].tick_params(direction="in", bottom=True, top=True, left=True, right=True)
    ax[2].plot(t, mz, lw=0.6)
    ax[2].set_ylabel("mz/Ms")
    ax[2].set_ylim([-1.1, 1.1])
    ax[2].tick_params(direction="in", bottom=True, top=True, left=True, right=True)
    ax[2].set_xlabel("time [s]")

    plt.figure(3)
    plt.title("unit vector lenght over time (for stability)")
    plt.plot(np.sqrt(mz ** 2 + mx ** 2 + my ** 2))
    plt.ylabel("Length unit vector")
    plt.xlabel("Timestep")

    plt.show()


if __name__ == "__main__":
    # let's run an example of the LLG solver for some IC.
    # NOTE: MUST ALWAYS START AT SOME ANGLE WHEN TEMPERATURE=0
    print("Starting trajectory calculation....")
    start = time.time()

    # defining relevant system parameters:
    Hext = np.array([-5e4, 0, 0])  # [A/m]
    alpha = 0.01  # SHOULD BE 0.01 FOR Cu!
    Ms = 1.27e6  # [A/m]
    K_surface = 0.5e-3  # J/m^2
    J = 0.1e12  # [A/m^2]
    thickness = 3e-9  # [m]
    width_x = 130e-9  # [m] need width_x > width_y >> thickness (we assume super flat ellipsoide)
    width_y = 70e-9  # [m]
    temperature = 3  # [K], note: need like 1e5 to see really in plot (just like MATLAB result)

    # initial direction free layer and fixed layer
    m0 = np.array([1, 0, 0])
    M3d = np.array([1, 0, 0])

    # which t points solve for, KEEP AS ARANGE (need same distance between points)!!
    t = np.arange(0, 5e-9, 1e-12)

    # solving the system
    mx, my, mz = LLG_solver(m0, t, Hext, alpha, Ms, J, thickness, width_x, width_y, temperature, M3d, K_surface, lambda t: 1)

    end = time.time()
    print(f"Code ran in {end - start} seconds")

    plotResult(mx, my, mz, m0, t)
