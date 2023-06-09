"""
Within this file, all the code necesarry for constructing phase diagrams will be implemented
"""
import istarmap
import os
from matplotlib.colors import ListedColormap
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from LLG_solver import *


def SingleLine(m0: np.array, t_points: np.array, alpha: float, Ms: float, thickness: float,
               width_x: float, width_y: float, temp: float, M3d: np.array, K_surf: float,
               skipLength: int, f_array: np.array, J:float, HextX: float, HextY: float, HextZ: float):
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

    resultDictionary = {}
    for index, freq in enumerate(f_array):

        # new current SHAPE for this run (amplitude was predefined):
        Jformula = lambda t: sin(2 * np.pi * freq * t)

        # solving the system
        HextFinal = np.array([HextX, HextY, HextZ])
        mx, my, mz = LLG_solver(m0, t_points, HextFinal, alpha, Ms, J, thickness, width_x,
                                width_y, temp, M3d, K_surf, Jformula)
        inspectMx, inspectMy, inspectMz = mx[skipLength:], my[skipLength:], mz[skipLength:]
        resultDictionary.update({J: [inspectMx, inspectMy, inspectMz]})

    analyzedResult = LineAnalysisFreq(resultDictionary)

    return analyzedResult


def LineAnalysisFreq(inputDictionary):
    """
    :param inputDictionary: The raw LLG results with as key the current and value [mx, my, mz]

    :returns: The categorized LLG results with as key the current and as value the maximum angle reached during
    precession
    """
    resultDictionary = {}
    for freq, values in inputDictionary.items():
        mx, my, mz = values[0], values[1], values[2]

        # calculate maximum angle w.r.t x axis reached:
        thetaX = np.arctan2(np.sqrt(my ** 2 + mz ** 2), mx)
        thetaXmax = np.max(thetaX) * 180 / np.pi

        resultDictionary.update({freq: thetaXmax})

    return resultDictionary


def TotalDiagram(
        m0: np.array, t_points: np.array, HextArray: np.array, alpha: float, Ms: float, thickness: float,
        width_x: float, width_y: float, temp: float, M3d: np.array, K_surf: float,
        skipLength: int, pool, f_array: np.array, J:float):
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
                            width_y, temp, M3d, K_surf, skipLength, f_array, J)
    HfieldArguments = [x for x in HextArray][0].tolist()

    print("mapping ...")
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


def PhaseDiagramPlot(X, Y, result):
    # fig3 = plt.figure(figsize=(6, 6))
    # plt.contour(result, alpha=1, colors='black', linewidths=6)
    # plt.xticks([])
    # plt.yticks([])
    fig2 = plt.figure(figsize=(6, 6))
    plt.pcolormesh(X, Y, result, cmap='inferno')
    plt.colorbar()

    # rewrite the labels of the axis to be in Tesla
    plt.yticks(ticks=plt.yticks()[0][1:-1], labels=np.round(constants.mu_0 * 1e3*np.array(plt.yticks()[0][1:-1]), 1))
    plt.ylabel("$μ_0 H [mT]$")
    plt.xlabel("$Freq [Hz]$")

    plt.show()


if __name__ == '__main__':
    # defining relevant system parameters:
    alpha = 0.01        # Damping parameter SHOULD BE 0.01 FOR Cu!
    Ms = 1.27e6         # [A/m]
    K_surface = 0.5e-3  # J/m^2
    d = 3e-9            # [m]
    width_x = 130e-9    # [m] need width_x > width_y >> thickness (we assume super flat ellipsoide)
    width_y = 70e-9     # [m]
    temperature = 3     # [K], note: need like 1e5 to see really in plot (just like MATLAB result)
    J = -0.1e-12        #[A/m^2]: current density amplitude

    # initial direction free layer and fixed layer
    m0 = np.array([1, 0, 0])
    M3d = polarToCartesian(1, np.pi / 2, 30 / 180 * np.pi)  # 30 degrees

    # which points solve for, KEEP AS ARANGE or linspace (need same distance between points)!!
    t = np.arange(0, 1e-9, 1e-12)
    gridSize = 30
    f_array = np.linspace(0, 1e10, gridSize)
    HextX = np.linspace(-9e3, -4.7e3, gridSize)  # [A/m]
    HextY = np.linspace(0, 0, gridSize)  # [A/m]
    HextZ = np.linspace(0, 0, gridSize)  # [A/m]
    HextArray = np.dstack([HextX, HextY, HextZ])

    # now don't need to skip, but look at all
    skipLength1 = 0
    inspectionT = t[skipLength1:]

    start = time.time()
    print("Starting calculation.....")

    try:
        pool = Pool(os.cpu_count() - 1)
        phaseDictionaryTuple = TotalDiagram(m0, t, HextArray, alpha, Ms, d, width_x, width_y, temperature, M3d, K_surface, skipLength1, pool, f_array,J)
    except Exception as e:
        print(e)
    finally:
        pool.close()

        end = time.time()
        print(f"process finished in {end - start} seconds")

        result = packagingForPlotting(phaseDictionaryTuple, gridSize)
        X, Y = np.meshgrid(f_array, HextX)  # Yes we also need to turn into meshgrid

        PhaseDiagramPlot(X, Y, result)
