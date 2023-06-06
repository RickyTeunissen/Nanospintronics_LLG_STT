
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
    eta = 1  # how add?
    pre = hbar / (2 * charge * d * mu0 * Ms)
    spin_transfer_torque = 1 / (1 + alpha ** 2) * current * (
            -pre * better_cross(m3d, better_cross(m3d, M3d)) - alpha * pre * better_cross(m3d, better_cross(m3d,
                                                                                                            better_cross(
                                                                                                                m3d,
                                                                                                                M3d))))
    # gyro_ratio / (mu0 * Ms) * eta * hbar / (2 * charge) * current / d * np.cross(m3d, np.cross(m3d, M3d))

    total_torque = spin_transfer_torque
    return total_torque

