"""
Within this file, all the code necisarry for constructing phase diagrams will be implemented
"""
import numpy as np
import LLG_solver as LLG


def SingleLine():
    # defining relevant system parameters:
    Hext = np.array([-5e5, 0, 0])  # [A/m]
    m0 = np.array([1, 0, 0])  # polarToCartesian(1, 0.49 * np.pi, 0.1)
    alpha = 0.1
    Ms = 1.27e6  # [A/m]
    d = 3e-9  # [m]
    area = 130e-9 * 70e-9
    M3d = np.array([0, 0, 1])  # polarToCartesian(1, 0.5*np.pi, 0)
    temperature = 300  # [K], note: need like 1e5 to see really in plot (just like MATLAB result)

    # which t points solve for, KEEP AS ARANGE (need same distance between points)!!
    t = np.arange(0, 5e-10, 1e-12)
    Jarray = np.linspace(-0.5e6, 0.6e6, 10)
    resultDictionary = {}
    skipLength = int(np.floor(len(t) * (2 / 3)))
    inspectionT = t[skipLength:]
    totLength = len(Jarray)
    for index, J in enumerate(Jarray):
        # solving the system
        mx, my, mz = LLG.LLG_solver(m0, t, Hext, alpha, Ms, J, d, area, temperature, M3d)
        inspectMx, inspectMy, inspectMz = mx[skipLength:], my[skipLength:], mz[skipLength:]
        resultDictionary.update({J: [inspectMx, inspectMy, inspectMz]})
        print(f"\r Working on step {index+1}/{totLength}", end="")

    print("\n finished")
    return resultDictionary


def LineAnalysis(inputDictionary):
    resultDictionary = {}
    for J, values in inputDictionary.items():
        mz = values[2]
        resultDictionary.update({J: np.average(mz)})
    return resultDictionary


result = SingleLine()
finalResult = LineAnalysis(result)
print(finalResult)