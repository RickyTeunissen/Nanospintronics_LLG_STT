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
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MultipleLocator

# Defining relevant physical constants.
mu0: float = constants.mu_0  # [N/A^2]
hbar: float = constants.hbar
me: float = constants.electron_mass  # [kg]
charge: float = constants.elementary_charge  # [C]
gyro_ratio: float = 2 * charge / (2 * me)  # [A/m]
kb: float = constants.Boltzmann  # [m^2 kg S^-2 K^-1]


def calc_effective_field(m3d: tuple, Hext: np.array, Ms: float, H_temp_array: np.array, current_time: float,
                         t_stepsize: float) -> tuple:
    """
    Calculates the effective field based on a given state of the system.
    Takes into account:
    - demag field
    - anisotropic field (NOT YET)
    - stochatsic field due to thermal fluctuations

    :param m3d: 3d tuple for the unit vector of the magnetizaiton (mx,my,mz)
    :param Hext: The externally applied field in [A/m]
    :param Ms: Saturation magnetizaion in [A/m]
    :param H_temp_array: array of random fields over time due to temperature (number timesteps)x3
    :param current_time: current time where the ODE solver is solving
    :param t_stepsize: stepsize in [s] between ODE solution points
    :return: Heff = (Heffx, Heffy, Heffz) = effective field in [A/m]
    """
    # Still add magnetocrystalline ani, maybe even T dep?:

    # we use thin films, these have a demagnetization field:
    H_demag = np.array([0, 0, -m3d[2] * Ms])

    # additional contribution due to temperature fluctuations first calculate in which step we are (s.t. in same t step
    # always pick same value if call this function > once)
    step_index = np.floor(current_time / t_stepsize).astype(int)
    H_temp = H_temp_array[step_index]

    Heff = H_demag + Hext + H_temp

    return Heff


def calc_total_spin_torque(current: float, m3d: np.array, M3d: np.array, Ms: float, d: float, area: float,
                           alpha: float):
    """
    Calcutes the total spin torque, includes:
    - spin transfer torque
    - spin pumpin (later)
    - Current-induced effective field (later? but often negligible)

    :param current: The current put through system [A]
    :param m3d: unit vector representing the direction of M in the free layer [mx,my,mz]
    :param M3d: unit vector representing the direction of M in the fixed layer [Mx,My,Mz]
    :param Ms: saturation magentization [A/m]
    :param d: thickness of sample [m]
    :param area: area of sample [m^2]
    :return: np.array([dmx, dmy, dmz]) due to spin torque
    """
    # if wanna add eta = ...
    # need to use: STT = -J/e*gyro*hbar/(2*Ms*mu0*d)*eta*1/(1+alpha**2)*(m cross m cross M - alpha*m cross M)

    ### ALLL TOTALLY WRONGGG #########
    # spin transfer torque contribution
    eta = 1 #how add?
    pre = hbar/(2*charge*d*mu0*Ms)
    spin_transfer_torque = 1 / (1 + alpha ** 2)*current*(-pre*np.cross(m3d, np.cross(m3d,M3d)) - alpha*pre*np.cross(m3d,np.cross(m3d, np.cross(m3d, M3d))))
    #gyro_ratio / (mu0 * Ms) * eta * hbar / (2 * charge) * current / d * np.cross(m3d, np.cross(m3d, M3d))

    total_torque = spin_transfer_torque
    return total_torque


def LLG(t, m3d: tuple, Hext: np.array, alpha: float, Ms: float, J: float, d: float, area: float, M3d: tuple,
        H_temp_array: np.array, t_stepsize):
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
    :param area: the cross sectional area of the pillar [m^2]
    :param M3d: tuple(Mx,My,Mz) = Unit vector, the magnetization of the fixed layer
    :param H_temp_array: array of random fields over time due to temperature (number timesteps)x3
    :return: (dmx,dmy,dmz)
    """

    # renormalize m due to stochastic field (is sketchy but better then nothing)
    m3d-= m3d - m3d/np.linalg.norm(m3d)

    #calculate preceission + damping:
    Heff = calc_effective_field(m3d, Hext, Ms, H_temp_array, t, t_stepsize)
    preces_damp = (-gyro_ratio * mu0 / (1 + alpha ** 2)) * (np.cross(m3d, Heff) + alpha * np.cross(m3d, np.cross(m3d, Heff)))

    # calculate contribution normal spin transfer torque (look my notes if wanna add eta)
    pre = -J/charge*gyro_ratio*hbar/(2*Ms*d)*1/(1+alpha**2)
    spinTorque = pre*(np.cross(m3d, np.cross(m3d, M3d)) - alpha*np.cross(m3d, M3d))

    dmx, dmy, dmz = preces_damp + spinTorque


    return dmx, dmy ,dmz


def LLG_solver(IC: np.array, t_points: np.array, Hext: np.array, alpha: float, Ms: float, J: float, d: float,
               area: float, temp: float, M3d: np.array) -> tuple:
    """
    Solves the LLG equation for given IC and parameters in the time range t_points

    :param IC: (mx0,my0,mz0) Initial magnetization of the free layer:
    :param t_points: Array of time points at which we wanna evaluate the solution
    :param Hext: tuple(Hx,Hy,Hz): the external magnetic field in [A/m]
    :param alpha: damping constant
    :param Ms: Saturation magnetization in [A/m]
    :param J: The current area density through the spin valve [A/m^2]
    :param d: The thickness of the free layer [m]
    :param area: the cross sectional area of the pillar [m^2]
    :param M3d: tuple(Mx,My,Mz) = Unit vector, the magnetization of the fixed layer
    :param temp: temperature of system [K]
    :return: unit vector m over time in
    """

    tspan = [t_points[0], t_points[-1]]  # ODE solver needs to know t bounds in advance
    t_step_size = t_points[1] - t_points[0]

    # precompute random field array due to temperature:
    H_temp_std = np.sqrt(2 * kb * alpha * temp / (mu0 * Ms ** 2 * d * area) / t_step_size)
    H_temp_arr = np.random.normal(0, H_temp_std, [t_points.shape[0], 3])  # gives array(meas points, 3) for all H terms

    # solve the ODE
    parameters = (Hext, alpha, Ms, J, d, area, M3d, H_temp_arr, t_step_size)
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
    f = plt.figure(1,figsize=(8,7))
    axf = f.add_subplot(projection="3d")
    axf.plot(mx, my, mz, "b-", lw=0.3)
    last_part = np.floor(mz.shape[0]*0.05).astype(int) # draw last 5% green to show e.g. stable orbit shape
    axf.plot(mx[-last_part:],my[-last_part:], mz[-last_part:], color = "lime", lw=2)
    axf.scatter(m0[0], m0[1], m0[2], color="red", lw=3) # startpoint
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
    plt.plot(np.sqrt(mz**2+mx**2+my**2))

    plt.show()


if __name__ == "__main__":
    # let's run an example of the LLG solver for some IC.
    # NOTE: MUST ALWAYS START AT SOME ANGLE WHEN TEMPERATURE=0
    print("Starting trajectory calculation....")
    start = time.time()

    # defining relevant system parameters:
    Hext = np.array([4.77e4, 0, 0])  # [A/m]
    m0 = np.array([1, 0, 0])  # polarToCartesian(1, 0.49 * np.pi, 0.1)
    alpha = 0.01  # SHOULD BE 0.01 FOR Cu!
    Ms = 1.27e6  # [A/m]
    J = 0.15e12 # [A/m^2]
    d = 3e-9  # [m]
    area = 130e-9 * 70e-9
    M3d = np.array([-1, 0, 0])  # polarToCartesian(1, 0.5*np.pi, 0)
    temperature = 3  # [K], note: need like 1e5 to see really in plot (just like MATLAB result)

    renormalize_counter = 100 # how often renormalize m3d

    # which t points solve for, KEEP AS ARANGE (need same distance between points)!!
    t = np.arange(0, 20e-9, 1e-12)

    # solving the system
    mx, my, mz = LLG_solver(m0, t, Hext, alpha, Ms, J, d, area, temperature, M3d)

    end = time.time()
    print(f"Code ran in {end - start} seconds")

    plotResult(mx, my, mz, m0, t)
