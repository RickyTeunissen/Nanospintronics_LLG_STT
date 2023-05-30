"""
Within this file, all the code necesarry for constructing phase diagrams will be implemented
"""
import time

import istarmap
import numpy as np
import LLG_solver as LLG
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import os
import scipy.constants as constants


def SingleLine(m0: np.array, t_points: np.array, alpha: float, Ms: float, thickness: float,
               width_x: float, width_y:float, temp: float, M3d: np.array, K_surf: float,
               useMemory: bool, skipLength: int, HextX: float, HextY: float, HextZ: float):
    """
        Calculates the effective field based over a range of applied currents.

        :param m0: 3d tuple for the unit vector of the magnetizaiton (mx,my,mz)
        :param t_points: Array of time points at which we wanna evaluate the solution
        :param alpha: damping constant
        :param Ms: Saturation magnetizaion in [A/m]
        :param thickness: thickness cylinder in [m]
        :param width_x: total width cylinder in the longest direction [m]
        :param width_y: total width cylinder in the smallest direction [m]
        :param temp: temperature of system [K]
        :param M3d: tuple(Mx,My,Mz) = Unit vector, the magnetization of the fixed layer
        :param K_surf: surface anisotropy that wants sys to go OOP [J/m]
        :param useMemory: boolean whether to use magnetic history
        :param skipLength: the amount of entries that must be skipped from the results
        :param HextX: The externally applied field in [A/m] in the x direction
        :param HextY: The externally applied field in [A/m] in the y direction
        :param HextZ: The externally applied field in [A/m] in the z direction

        :returns: Dictionary of the analyzed results, categorized by state.
    """

    resultDictionary = {}
    for index, J in enumerate(Jarray):
        # solving the system
        if useMemory & index > 0:
            m0 = np.array([mx[-1], my[-1], mz[-1]])
        HextFinal = np.array([HextX, HextY, HextZ])
        mx, my, mz = LLG.LLG_solver(m0, t_points, HextFinal, alpha, Ms, J, thickness, width_x, width_y, temp, M3d, K_surf)
        inspectMx, inspectMy, inspectMz = mx[skipLength:], my[skipLength:], mz[skipLength:]
        resultDictionary.update({J: [inspectMx, inspectMy, inspectMz]})

    analyzedResult = LineAnalysis(resultDictionary)
    return analyzedResult


def LineAnalysis(inputDictionary):
    resultDictionary = {}
    for J, values in inputDictionary.items():
        mz = values[2]
        z = np.average(mz).round(2)
        mx = values[0]
        x = np.average(mx).round(2)
        conditionListz = [z > 0.1 or z < -0.1, -0.1 <= z <= 0.1]  # OPP, IPP
        conditionListx = [x > 0.9, -0.9 <= x < 0.9, x < -0.9]  # AntiParallel, Precession, Parallel
        stateConditionList = [conditionListx[0], conditionListx[1] and conditionListz[1], conditionListx[1] and conditionListz[0], conditionListx[2]]  # AP, IPP, OPP, P

        choiceList = [1, 2, 3, 4]  # P, IPP, OPP, AP
        categorizedValues = np.select(stateConditionList, choiceList)
        resultDictionary.update({J: categorizedValues})
    return resultDictionary

def TotalDiagram(
        m0: np.array, t_points: np.array, HextArray: np.array, alpha: float, Ms: float, thickness: float,
        width_x: float, width_y: float, temp: float, M3d: np.array, K_surf: float, useMemory: bool, skipLength: int, pool):
    """
        Calculates the effective field based over a range of applied currents.

        :param m0: 3d tuple for the unit vector of the magnetizaiton (mx,my,mz)
        :param t_points: Array of time points at which we wanna evaluate the solution
        :param HextArray: The externally applied field in [A/m] in the form [[x,y,z],[],...]
        :param alpha: damping constant
        :param Ms: Saturation magnetizaion in [A/m]
        :param thickness: thickness cylinder in [m]
        :param width_x: total width cylinder in the longest direction [m]
        :param width_y: total width cylinder in the smallest direction [m]
        :param temp: temperature of system [K]
        :param M3d: tuple(Mx,My,Mz) = Unit vector, the magnetization of the fixed layer
        :param K_surf: surface anisotropy that wants sys to go OOP [J/m]
        :param useMemory: boolean whether to use magnetic history
        :param skipLength: the amount of entries that must be skipped from the results
        :param pool: The multiprocessor pool

        :returns: A tuple of the complete phase diagram, with each entry 1 sweep over the current.
    """

    partialSweepH = partial(SingleLine, m0, t_points, alpha, Ms, thickness, width_x,
                            width_y, temp, M3d, K_surf, useMemory, skipLength)
    HfieldArguments = [x for x in HextArray][0].tolist()

    print("mapping ...")
    resultsIterable = tqdm(pool.istarmap(partialSweepH, HfieldArguments), total=len(HfieldArguments))
    print("running ...")
    results = tuple(resultsIterable)
    print("done")
    return results


def packagingForPlotting(inputList, gridSize: int):
    stateArray = np.zeros((gridSize, gridSize))
    for index, currentDictionary in enumerate(inputList):
        currentArray = []
        for current, state in currentDictionary.items():
            currentArray.append(state.tolist())
        stateArray[index] = currentArray
    return stateArray


# defining relevant system parameters:
alpha = 0.01  # SHOULD BE 0.01 FOR Cu!
Ms = 1.27e6  # [A/m]
K_surface = 0.5e-3  # J/m^2
d = 3e-9  # [m]
width_x = 130e-9  # [m] need width_x > width_y >> thickness (we assume super flat ellipsoide)
width_y = 70e-9  # [m]
temperature = 3  # [K], note: need like 1e5 to see really in plot (just like MATLAB result)

# initial direction free layer and fixed layer
m0 = np.array([1, 0, 0])
M3d = np.array([1, 0, 0])

# which t points solve for, KEEP AS ARANGE (need same distance between points)!!
t = np.arange(0, 7e-9, 5e-12)
gridSize = 15
Jarray = np.linspace(-0.5e12, 0.5e12, gridSize)
HextX = np.linspace(-5.5e4, 5.5e4, gridSize)  # [A/m]
HextY = np.linspace(0, 0, gridSize)  # [A/m]
HextZ = np.linspace(0, 0, gridSize)  # [A/m]
HextArray = np.dstack([HextX, HextY, HextZ])

# Skip a fraction of the time, to ensure we take the more steady state ish result
skipLength1 = int(np.floor(len(t) * (2 / 4)))
inspectionT = t[skipLength1:]


if __name__ == '__main__':
    start = time.time()
    print("Starting calculation.....")

    try:
        pool = Pool(os.cpu_count())
        phaseDictionary = TotalDiagram(np.array([1, 0, 0]), t, HextArray, alpha, Ms, d, width_x,
                     width_y, temperature, M3d, K_surface, False, skipLength1, pool)
    finally:
        pool.close()

        end = time.time()
        print(f"process finished in {end-start} seconds")

        result = packagingForPlotting(phaseDictionary, gridSize)
        X, Y = np.meshgrid(Jarray, HextX)  # Yes we also need to turn into meshgrid
        flat_list = [item for sublist in result for item in sublist]
        levels = len(set(flat_list)) - 1

        # colors = ['LightGray', 'Gray', 'DarkGray', 'DimGray']
        # colors = ['LightCoral', 'IndianRed', 'FireBrick', 'DarkRed']
        # colors = ['DodgerBlue', 'LimeGreen', 'Goldenrod', 'MediumPurple']
        colors = ['#3fe522', '#6c38cc', '#d661ad', '#cb6934']
        # colors = ['#dfd562', '#b0e16b', '#bdd73c', '#dcaf56']


        cmap = ListedColormap(colors)

        fig3 = plt.figure(figsize=(6, 6))
        plt.contour(result, levels=[1, 2, 3, 4], alpha=1, colors='black', linewidths=6)
        plt.xticks([])
        plt.yticks([])

        fig2 = plt.figure(figsize=(6, 6))
        plt.pcolormesh(X, Y, result, cmap=cmap, )
        plt.colorbar()
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=colors[0], label='AP'),
            plt.Rectangle((0, 0), 1, 1, color=colors[1], label='IPP'),
            plt.Rectangle((0, 0), 1, 1, color=colors[2], label='OPP'),
            plt.Rectangle((0, 0), 1, 1, color=colors[3], label='P')
        ]

        # Add the legend
        plt.legend(handles=legend_elements)

        # rewrite the labels of the axis to be in Tesla
        plt.yticks(ticks=plt.yticks()[0][1:-1], labels=np.round(constants.mu_0 * np.array(plt.yticks()[0][1:-1]), 2))
        plt.ylabel("$μ_0 H [T]$")
        plt.xlabel("$J [A/m^2]$")

        plt.show()