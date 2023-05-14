"""
Within this file I will show a few examples of solivng ODE (systems) with the scipy.solve_ipv function:
- solve a simple exp decay = 1D ODE
- Show how to interpolate data which is faster then solving the ODE at more points, but less reliable
- Solve a 3D ODE system: the lorenz oscillator
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d



# In this example we will look at solving simple exponential decay: dxdt = -a x
# As input need some t variable as first variable (even if not explicitly present in the ODE, the solver need some
# variable to store the timestep), then the state: for which we wanna solve, then other parameters
def dxdt(t, x, a):
    """
    In this example we will look at solving simple exponential decay: dxdt = -a x

    :param t: 1D array of ind variable
    :param x: some ND array of the N dimensional ODE
    :param a: any other variables we like to supply

    :return: the right hand side to dxdt =....
    """
    return a*x

def lorenz(t, state, sigma, beta, rho):
    """
    Equation for a lorenz oscillator, again of the form d(x,y,z)dt = lorenzoscilatoor(...)
    Note that these are actually 3 coupled ODE's
    :param t: 1D array
    :param state: array (x,y,z)
    :param sigma,beta,rho: other relevant para
    :return: tuple(dx,dy,dz) = Right hand side of d(x,y,z)dt
    """
    x,y,z = state
    dx = sigma*(y-x)
    dy = x*(rho-z)-y
    dz = x*y-beta*z

    return (dx,dy,dz)


if __name__ == "__main__":
    figure, ax = plt.subplots(2, 1, squeeze=False)

    ################ First solve a 1D ODE #####################
    tspan = np.array([0, 20]) # tspan = start+end t, instead can also use t_val if wanna define t_pos where solve
    x0 = np.array(5)
    a0 = -0.5
    ODE_sol = solve_ivp(dxdt, t_span=tspan, y0=[5], args=(a0, ))
    x_sol = ODE_sol.y.transpose()[:,0]  #note how if we have a 1D system, the array might be of a wrong shape
    t_sol_at = ODE_sol.t

    print("First a general ouput we get for Solve_IVP:")
    print(ODE_sol)

    ax[0, 0].plot(t_sol_at, x_sol, "o--")
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("X(t)")

    # to make the plotter of better quality either we can manually supply t-points were to solve the system (see next
    # example) or smarter: we can use numpy to interpolate the values in between -> Probabl DO NOT DO!!!
    interpolation_function = interp1d(t_sol_at, x_sol, kind="cubic")
    new_t_pos = np.linspace(0, 20, 50)
    new_x = interpolation_function(new_t_pos)
    ax[1, 0].plot(new_t_pos, new_x, "o--")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("X(t)")


    ###################### NOW solve a coupled = multidimensional ODE ####################
    sigma = 10.0
    beta = 8.0/3.0
    rho = 28.0
    p = (sigma, beta, rho) # parameters for the system

    y0 = [1.0, 1.0, 1.0]     # IC system
    t_span = [0, 40]         # must ALWAYS give t span
    t = np.arange(0.0, 40.0, 0.01) # now define exact t points want solution, must lay within t_span

    lorenz_sol = solve_ivp(lorenz, t_span, t_eval=t, y0=y0, args=p)

    print("note how in 3d solutions are given:")
    print(lorenz_sol.y)
    x_sol = lorenz_sol.y[0, :]
    y_sol = lorenz_sol.y[1, :]
    z_sol = lorenz_sol.y[2, :]

    f = plt.figure(2)
    axf = f.add_subplot(projection='3d')
    axf.plot(x_sol,y_sol,z_sol, lw =0.2)
    axf.set_xlabel("x")
    axf.set_ylabel("y")
    axf.set_zlabel("z")
    axf.set_title("Ricky is the best!")

    plt.show()

