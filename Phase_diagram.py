"""
Within this file, all the code necisarry for constructing phase diagrams will be implemented
"""
import numpy as np
import LLG_solver

Hext = np.array([0.2e6, 0, 0])  # [A/m]
m0 = LLG_solver.polarToCartesian(1, 0.4*np.pi, 0)
alpha = 0.1
Ms = 1.27e6  # [A/m]
J = 0.0001  # [???] some form of A/m^2
d = 1e-9  # [m]
M3d = LLG_solver.polarToCartesian(1, 0, 0)
t = np.arange(0, 2.5e-9, 1e-12)
mx, my, mz = LLG_solver.LLG_solver(m0, t, Hext, alpha, Ms, J, d, M3d)

LLG_solver.plotResult(mx, my, mz, m0, t)