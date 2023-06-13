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
                f_array: np.array, J:float, HextX: float, HextY: float, HextZ: float):
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
        :param Jarray: an array of the applied current values
        :param frequency: The frequency of the AC oscillations
        :param HextX: The externally applied field in [A/m] in the x direction
        :param HextY: The externally applied field in [A/m] in the y direction
        :param HextZ: The externally applied field in [A/m] in the z direction

        :returns: Dictionary of the analyzed results, categorized by state.
    """

    resultDictionary = {}
    for freq in f_array:

        # new current SHAPE for this run (amplitude was predefined):
        Jformula = lambda t: sin(2 * np.pi * freq * t)

        # solving the system
        HextFinal = np.array([HextX, HextY, HextZ])
        mx, my, mz = LLG_solver(m0, t_points, HextFinal, alpha, Ms, J, thickness, width_x,
                                width_y, temp, M3d, K_surf, Jformula)

        resultDictionary.update({freq: [mx, my, mz]})

    analyzedResult = LineAnalysisFreq(resultDictionary, M3d)
    return analyzedResult


def LineAnalysisFreq(inputDictionary, M3d):
    """
    :param inputDictionary: The raw LLG results with as key the current and value [mx, my, mz]

    :returns: The categorized LLG results with as key the current and as value the maximum angle reached during
    precession
    """
    resultDictionary = {}
    for freq, values in inputDictionary.items():
        mx, my, mz = values

        # calculate maximum angle w.r.t x axis reached:
        mArray = np.array(values).T
        dotproduct = np.sum(mArray * M3d, axis=1)
        magnitudem = np.linalg.norm(mArray, axis=1)
        magnitudeM = np.linalg.norm(M3d)
        angles = np.arccos(dotproduct / (magnitudeM * magnitudem))
        thetaXmax = np.max(angles) * 180 / np.pi

        resultDictionary.update({freq: thetaXmax})

    return resultDictionary


def TotalDiagram(
        m0: np.array, t_points: np.array, HextArray: np.array, alpha: float, Ms: float, thickness: float,
        width_x: float, width_y: float, temp: float, M3d: np.array, K_surf: float, pool, f_array: np.array, J:float):
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
        :param pool: The multiprocessor pool
        :param Jarray: an array of the applied current values
        :param frequency: The frequency of the AC oscillations

        :returns: A tuple of the complete phase diagram, with each entry the dictionary over the current.
    """

    partialSweepH = partial(SingleLine, m0, t_points, alpha, Ms, thickness, width_x,
                            width_y, temp, M3d, K_surf, f_array, J)
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

def onclick(event, fig, ax, line_x, line_y):
    if event.inaxes == ax:
        x = event.xdata
        y = event.ydata

        line_x.append(x)
        line_y.append(y)

        ax.plot(x, y, 'ro')
        fig.canvas.draw()


def PhaseDiagramPlot(X, Y, result):

    fig2, ax = plt.subplots(figsize=(6, 6))
    plt.pcolormesh(X, Y, result, cmap='inferno', picker=5)
    plt.colorbar()

    # display value obtained within the cell (comment away if not wanted:
    # for (x, y), value in np.ndenumerate(result):
    #     xpos = X[x,y]
    #     ypos = Y[x,y]
    #     plt.text(xpos, ypos, f"{value:.0f}", va="center", ha="center",color = "grey" )

    # also plot the kittel equation for comparison
    # H_used = np.abs(np.transpose(Y)[0])
    # f_kittel_res = gyro_co*np.sqrt(H_used*(H_used+Ms))/(2*np.pi)
    # plt.plot(f_kittel_res, H_used, lw=1, color="white")

    # rewrite the labels of the axis to be in Tesla
    plt.yticks(ticks=plt.yticks()[0][1:-1], labels=np.round(constants.mu_0 * 1e3*np.array(plt.yticks()[0][1:-1]), 1))
    plt.ylabel("$μ_0 H [mT]$")
    plt.xlabel("$Freq [Hz]$")

    line_x = []
    line_y = []

    fig2.canvas.callbacks.connect('button_press_event', lambda event: onclick(event, fig2, ax, line_x, line_y))
    plt.show()
    print("Line x-positions:", line_x)
    print("Line y-positions:", line_y)


if __name__ == '__main__':
    # defining relevant system parameters:
    alpha = 0.01        # Damping parameter SHOULD BE 0.01 FOR Cu!
    Ms = 1.27e6         # [A/m]
    K_surface = 0.5e-3  # J/m^2
    d = 3e-9            # [m]
    width_x = 130e-9    # [m] need width_x > width_y >> thickness (we assume super flat ellipsoide)
    width_y = 70e-9     # [m]
    temperature = 3     # [K], note: need like 1e5 to see really in plot (just like MATLAB result)
    J = -1e11        #[A/m^2]: current density amplitude

    # initial direction free layer and fixed layer
    m0 = np.array([1, 0, 0])
    M3d = polarToCartesian(1, np.pi / 2, 30 / 180 * np.pi)  # 30 degrees w.r.t. long axis

    # which points solve for, KEEP AS ARANGE or linspace (need same distance between points)!!
    t = np.arange(0, 1e-9, 5e-12)
    gridSize = 100
    f_array = np.linspace(0.5e9, 8e9, gridSize)
    HextX = np.linspace(-6e3, 50e3, gridSize)  # [A/m]
    HextY = np.linspace(0, 0, gridSize)  # [A/m]
    HextZ = np.linspace(0, 0, gridSize)  # [A/m]
    HextArray = np.dstack([HextX, HextY, HextZ])

    start = time.time()
    print("Starting calculation.....")

    try:
        pool = Pool(os.cpu_count() - 1)
        phaseDictionaryTuple = TotalDiagram(m0, t, HextArray, alpha, Ms, d, width_x, width_y, temperature, M3d, K_surface, pool, f_array,J)
    except Exception as e:
        print(e)    # catch an exception in case anything happens
    finally:
        pool.close() # Be sure to in end close the pool!!! (computer won't like it else):

        end = time.time()
        print(f"process finished in {end - start} seconds")

        result = packagingForPlotting(phaseDictionaryTuple, gridSize)
        X, Y = np.meshgrid(f_array, HextX)  # Yes we also need to turn into meshgrid

        PhaseDiagramPlot(X, Y, result)
