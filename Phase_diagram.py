"""
Within this file, all the code necesarry for constructing phase diagrams will be implemented
"""
import time

import istarmap
import numpy as np
import LLG_solver as LLG
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from math import sin
import os
import scipy.constants as constants
import winsound

def SingleLine(m0: np.array, t_points: np.array, alpha: float, Ms: float, thickness: float,
               width_x: float, width_y:float, temp: float, M3d: np.array, K_surf: float,
               useAC: bool, skipLength: int, Jarray: np.array, frequency: float, HexNorm: np.array, HextX: float, HextY: float, HextZ: float):
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
        :param useAC: boolean whether to use AC current
        :param skipLength: the amount of entries that must be skipped from the results
        :param Jarray: an array of the applied current values
        :param frequency: The frequency of the AC oscillations
        :param HextX: The externally applied field in [A/m] in the x direction
        :param HextY: The externally applied field in [A/m] in the y direction
        :param HextZ: The externally applied field in [A/m] in the z direction

        :returns: Dictionary of the analyzed results, categorized by state.
    """

    if useAC:
        Jformula = lambda t: sin(frequency * t)
    else:
        Jformula = lambda t: 1

    resultDictionary = {}
    for index, J in enumerate(Jarray):
        # solving the system
        HextFinal = np.array([HextX, HextY, HextZ])
        mx, my, mz = LLG.LLG_solver(m0, t_points, HextFinal, alpha, Ms, J, thickness, width_x,
                                    width_y, temp, M3d, K_surf, Jformula)
        inspectMx, inspectMy, inspectMz = mx[skipLength:], my[skipLength:], mz[skipLength:]
        resultDictionary.update({J: [inspectMx, inspectMy, inspectMz]})

    if useAC:
        analyzedResult = LineAnalysisAC(resultDictionary)
    else:
        analyzedResult = LineAnalysis(resultDictionary, M3d, HexNorm)

    return analyzedResult


def LineAnalysisAC(inputDictionary):
    """
        Takes the max of mx.

        :param inputDictionary: The raw LLG results with as key the current and value [mx, my, mz]

        :returns: The categorized LLG results with as key the current and value [state]
    """
    resultDictionary = {}
    for J, values in inputDictionary.items():
        mx = values[0]
        my = values[1]
        mz = values[2]
        thetaX = np.arctan2(np.sqrt(my**2 + mz**2), mx)
        thetaXmax = np.max(thetaX)*180/np.pi
        resultDictionary.update({J: thetaXmax})
    return resultDictionary


def LineAnalysis(inputDictionary, M3d: np.array, HexNorm: np.array):
    """
        Takes the average of mx and mz and categorizes the state based on those averages.

        :param inputDictionary: The raw LLG results with as key the current and value [mx, my, mz]

        :returns: The categorized LLG results with as key the current and value [state]
    """
    resultDictionary = {}
    for J, values in inputDictionary.items():
        (mx, my, mz) = (values[0], values[1], values[2])
        y = np.average(my).round(3)
        x = np.average(mx).round(3)
        (stdx, stdy, stdz) = (np.std(mx), np.std(my), np.std(mz))
        conditionIPP = stdx > 0.1 and stdy > 0.1 and stdz > 0.1
        conditionOPP = stdx > 0.1 and stdy > 0.1 and stdz < 0.1
        conditionStableP = not conditionIPP and not conditionOPP and x > 0.95 and abs(y) < 0.5*abs(M3d[1])
        conditionStableAP = not conditionIPP and not conditionOPP and x < -0.95 and abs(y) < 0.5*abs(M3d[1])
        pinnedLayerAligned = not conditionStableAP and not conditionStableP
        stateConditionList = [conditionStableP, conditionIPP, conditionOPP, conditionStableAP, pinnedLayerAligned]

        choiceList = [0, 1, 2, 3, 4]  # P, IPP, OPP, AP, FLA
        categorizedValues = np.select(stateConditionList, choiceList)
        resultDictionary.update({J: categorizedValues})
    return resultDictionary

def TotalDiagram(
        m0: np.array, t_points: np.array, HextArray: np.array, alpha: float, Ms: float, thickness: float,
        width_x: float, width_y: float, temp: float, M3d: np.array, K_surf: float, useAC: bool,
        skipLength: int, pool, Jarray: np.array, frequency: float, HexNorm: np.array):
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
        :param useAC: boolean whether to use AC current
        :param skipLength: the amount of entries that must be skipped from the results
        :param pool: The multiprocessor pool
        :param Jarray: an array of the applied current values
        :param frequency: The frequency of the AC oscillations

        :returns: A tuple of the complete phase diagram, with each entry the dictionary over the current.
    """

    partialSweepH = partial(SingleLine, m0, t_points, alpha, Ms, thickness, width_x,
                            width_y, temp, M3d, K_surf, useAC, skipLength, Jarray, frequency, HexNorm)
    HfieldArguments = [x for x in HextArray][0].tolist()

    resultsIterable = tqdm(pool.istarmap(partialSweepH, HfieldArguments), total=len(HfieldArguments))
    results = tuple(resultsIterable)
    print("done")
    return results


def packagingForPlotting(inputDictionaryTuple: tuple, gridSize: int):
    """
        Unpacks the tuple of dictionaries into a list of lists.

        :param inputDictionaryTuple: A tuple with each entry a dictionary of the current at 1 field strength
        Each dictionary has as key the current and value the state [state]
        :param gridSize: The gridsize of the phase diagram

        :returns: A list of lists, with each entry a sweep over the current at a specific field strength.
    """

    stateArray = np.zeros((gridSize, gridSize))
    for index, currentDictionary in enumerate(inputDictionaryTuple):
        currentArray = []
        for current, state in currentDictionary.items():
            currentArray.append(state.tolist())
        stateArray[index] = currentArray
    return stateArray


def PhaseDiagramPlotDC(X, Y):

    colors = ['#3fe522', '#6c38cc', '#d661ad', '#cb6934', 'black']

    cmap = ListedColormap(colors)

    fig3 = plt.figure(figsize=(6, 6))
    plt.contour(result, levels=[1, 2, 3, 4, 5], alpha=1, colors='black', linewidths=6)
    plt.xticks([])
    plt.yticks([])

    fig2 = plt.figure(figsize=(6, 6))
    plt.pcolormesh(X, Y, result, cmap=cmap, )
    plt.colorbar()
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=colors[0], label='P'),
        plt.Rectangle((0, 0), 1, 1, color=colors[1], label='IPP'),
        plt.Rectangle((0, 0), 1, 1, color=colors[2], label='OPP'),
        plt.Rectangle((0, 0), 1, 1, color=colors[3], label='AP'),
        plt.Rectangle((0, 0), 1, 1, color=colors[4], label='FLA'),
    ]

    # Add the legend
    plt.legend(handles=legend_elements, loc='upper right')

    # rewrite the labels of the axis to be in Tesla
    plt.yticks(ticks=plt.yticks()[0][1:-1], labels=np.round(constants.mu_0 * np.array(plt.yticks()[0][1:-1])*1e3, 2))
    plt.ylabel("$μ_0 H [mT]$")
    plt.xlabel("$J [A/m^2]$")

    plt.show()


def PhaseDiagramPlotAC(X, Y, title: str):

    fig3 = plt.figure(figsize=(6, 6))
    plt.contour(result, alpha=1, colors='black', linewidths=6)
    plt.xticks([])
    plt.yticks([])

    fig2 = plt.figure(figsize=(6, 6))
    plt.pcolormesh(X, Y, result, cmap='hot')
    plt.colorbar()

    # rewrite the labels of the axis to be in Tesla
    plt.yticks(ticks=plt.yticks()[0][1:-1], labels=np.round(constants.mu_0 * np.array(plt.yticks()[0][1:-1])*1e3, 0))
    plt.ylabel("$μ_0 H [mT]$")
    plt.xlabel("$J [A/m^2]$")
    plt.title(title)

    plt.show()


if __name__ == '__main__':
    # defining relevant system parameters:
    alpha = 0.01  # SHOULD BE 0.01 FOR Cu!
    Ms = 1.27e6  # [A/m]
    K_surface = 0.5e-3  # J/m^2
    d = 3e-9  # [m]
    width_x = 130e-9  # [m] need width_x > width_y >> thickness (we assume super flat ellipsoide)
    width_y = 70e-9  # [m]
    temperature = 3  # [K]

    # initial direction free layer and fixed layer
    m0 = np.array([1, 0, 0])
    M3d = LLG.polarToCartesian(1, np.pi/2, np.pi/6)

    # which t points solve for, KEEP AS ARANGE (need same distance between points)!!
    t = np.arange(0, 5e-9, 5e-12)
    gridSize = 200
    Jarray = np.linspace(-0.15e12, 0.15e12, gridSize)
    UseAcJ = False  #True = run AC special plotting too, False = good ol phase diagram
    frequency = 5.83e8
    HexNorm = LLG.polarToCartesian(1, np.pi/2, 0)
    HexSize = np.linspace(-1.1e4, 1.1e4, gridSize)
    HextX = HexSize * HexNorm[0]  # [A/m]
    HextY = HexSize * HexNorm[1]  # [A/m]
    HextZ = HexSize * HexNorm[2]  # [A/m]
    HextArray = np.dstack([HextX, HextY, HextZ])

    # Skip a fraction of the time, to ensure we take the more steady state ish result
    skipLength1 = int(np.floor(len(t) * (3 / 5)))
    inspectionT = t[skipLength1:]

    start = time.time()
    print("Starting calculation.....")

    try:
        cpu_available = os.cpu_count() - 1  # use all but 1 to ensure can still type/etc. (it is possible to use all)
        print(f"Using {cpu_available} cores to simulate")
        pool = Pool(cpu_available)

        phaseDictionaryTuple = TotalDiagram(m0, t, HextArray, alpha, Ms, d, width_x,
                     width_y, temperature, M3d, K_surface, UseAcJ, skipLength1, pool, Jarray, frequency, HexNorm)
    finally:
        pool.close()
        end = time.time()
        print(f"process finished in {end-start} seconds")

        result = packagingForPlotting(phaseDictionaryTuple, gridSize)
        X, Y = np.meshgrid(Jarray, HextX)  # Yes we also need to turn into meshgrid

        winsound.Beep(500, 2000)
        winsound.Beep(800, 1000)
        winsound.Beep(1300, 500)

        if UseAcJ:
            PhaseDiagramPlotAC(X, Y, str(frequency))
        else:
            PhaseDiagramPlotDC(X, Y)


