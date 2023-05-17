"""
The code in this file can be run to create a single moke loop,
the idea is as follows:
for a given set of system parameters the LLG solver is run for a "long" time until the magnetizaiton is deemed to have
stabalized. Then it is run for a bit longer s.t. for this period of time a time average of the z component of the
magnetization can be made over various oscillations, this way the character can be seen:
-<mz> = -1: AP
-<mz> = +1: P
-<mz> = 0: preceissing + IP
-  -1<<mz>1: and !=0: preceising + OOP at some angle

now 1 of the parameters is increased in a small step and the procedure is repeated where the IC for the next calculation
are chosen to be the final conditions of the previous one.
"""
from LLG_solver import LLG_solver
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    # first slowly increase J
    #for Current in range():

    Hext_z = np.arange(-1e7,1e7,1e6)
    mz_av = np.zeros(np.shape(Hext_z)[0])
    for step in range(np.shape(Hext_z)[0]):
        print(f"\r Working on step {step}/{np.shape(Hext_z)[0]}", end="")
        #Run LLG solver, must run at least for .... (STIL LOOK INTO)
        Hext = np.array([0, 10e6, Hext_z[step]])  # external field in the y direction
        m0 = np.sqrt(1 / 2) * np.array([1, 1, 0])
        t = np.arange(0, 5e-9, 1e-11)
        mx, my, mz = LLG_solver(m0, t, Hext)

        #Taka timeavergae of final part where aready have stabalized
        stable_mz_av = np.mean(mz[-200:-1].copy())
        mz_av[step] = stable_mz_av


    # now repeat but slowly decrease:
    Hext_z_2 = np.arange(1e7, -1e7, -1e6)
    mz_av_2 = np.zeros(np.shape(Hext_z)[0])
    for step in range(np.shape(Hext_z)[0]):
        print(f"\r Working on step {step}/{np.shape(Hext_z_2)[0]}", end="")
        # Run LLG solver, must run at least for .... (STIL LOOK INTO)
        Hext = np.array([0, 10e6, Hext_z_2[step]])  # external field in the y direction
        m0 = np.sqrt(1 / 2) * np.array([1, 1, 0])
        t = np.arange(0, 5e-9, 1e-11)
        mx, my, mz = LLG_solver(m0, t, Hext)

        # Taka timeavergae of final part where aready have stabalized
        stable_mz_av = np.mean(mz[-200:-1].copy())
        mz_av_2[step] = stable_mz_av

    print(mz_av)

    plt.plot(Hext_z, mz_av, Hext_z_2,mz_av_2)
    plt.show()