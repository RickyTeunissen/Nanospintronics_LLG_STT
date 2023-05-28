"""
Within this file, all the code necesarry for constructing phase diagrams will be implemented
"""
import time

import istarmap
import numpy as np
import LLG_solver as LLG
import matplotlib.pyplot as plt
import scipy.constants as constants
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import os

def SingleLine(m0: np.array, t_points: np.array, Hext: float, alpha: float, Ms: float, thickness: float,
               width_x: float, width_y: float, temp: float, M3d: np.array, K_surf: float, useMemory: bool,
               skipLength: int):
    resultDictionary = {}

    for index, J in enumerate(Jarray):
        # solving the system
        if useMemory & index > 0:
            m0 = np.array([mx[-1], my[-1], mz[-1]])
        HextFinal = np.array([Hext, 0, 0])
        mx, my, mz = LLG.LLG_solver(m0, t_points, HextFinal, alpha, Ms, J, thickness, width_x, width_y, temp, M3d,
                                    K_surf)
        inspectMx, inspectMy, inspectMz = mx[skipLength:], my[skipLength:], mz[skipLength:]
        resultDictionary.update({J: [inspectMx, inspectMy, inspectMz]})
    return resultDictionary


def LineAnalysis(inputDictionary):
    resultDictionary = {}
    for J, values in inputDictionary.items():
        mz = values[2]
        z = np.average(mz).round(2)
        mx = values[0]
        x = np.average(mx).round(2)
        conditionListz = [z > 0.1 or z < -0.1, -0.1 <= z <= 0.1]  # OPP, IPP
        conditionListx = [x > 0.9, -0.9 <= x < 0.9, x < -0.9]  # AntiParallel, Precession, Parallel
        stateConditionList = [conditionListx[0], conditionListx[1] and conditionListz[1],
                              conditionListx[1] and conditionListz[0], conditionListx[2]]  # AP, IPP, OPP, P

        choiceList = [1, 2, 3, 4]  # P, IPP, OPP, AP
        categorizedValues = np.select(stateConditionList, choiceList)
        resultDictionary.update({J: categorizedValues})
    return resultDictionary


def SweepH(m0: np.array, t_points: np.array, alpha: float, Ms: float, thickness: float, width_x: float,
           width_y: float, temp: float, M3d: np.array, K_surf: float, useMemory: bool, skipLength: int,
           totLength: float,
           HextX: float, HextY: float, HextZ: float):
    result = SingleLine(
        m0, t_points, HextX, alpha, Ms, thickness, width_x, width_y, temp, M3d, K_surf, useMemory, skipLength)
    analyzedResult = LineAnalysis(result)
    # phaseDiagramDictionary.update({Hext[0]: analyzedResult})
    return analyzedResult


def TotalDiagram(
        m0: np.array, t_points: np.array, HextArray: np.array, alpha: float, Ms: float, thickness: float,
        width_x: float, width_y: float, temp: float, M3d: np.array, K_surf: float, useMemory: bool, skipLength: int,
        pool):
    totLength = len(HextArray[0])
    partialSweepH = partial(SweepH, m0, t_points, alpha, Ms, thickness, width_x,
                            width_y, temp, M3d, K_surf, useMemory, skipLength, totLength)
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
gridSize = 10
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
        print(f"process finished in {end - start} seconds")

        result = packagingForPlotting(phaseDictionary, gridSize)
        X, Y = np.meshgrid(Jarray, HextX)  # Yes we also need to turn into meshgrid
        flat_list = [item for sublist in result for item in sublist]
        levels = len(set(flat_list)) - 1

        # colors = ['LightGray', 'Gray', 'DarkGray', 'DimGray']
        # colors = ['LightCoral', 'IndianRed', 'FireBrick', 'DarkRed']
        colors = ['DodgerBlue', 'LimeGreen', 'Goldenrod', 'MediumPurple']

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
        plt.yticks(ticks=plt.yticks()[0][1:-1], labels= np.round(constants.mu_0 * np.array(plt.yticks()[0][1:-1]), 2))
        plt.ylabel("$μ_0 H [T]$")
        plt.xlabel("$J [A/m^2]$")

        plt.show()

# phaseDictionary = TotalDiagram(
#    np.array([1, 0, 0]), t, HextArray, alpha, Ms, d, width_x, width_y, temperature, M3d, K_surface, False, skipLength1)
# result = packagingForPlotting(phaseDictionary, gridSize)

# X, Y = np.meshgrid(Jarray, HextX)  # Yes we also need to turn into meshgrid

# fig2 = plt.figure(figsize=(6, 6))
# plt.pcolormesh(X, Y, result, cmap=cm.Blues, )
# plt.colorbar()

# plt.show()
