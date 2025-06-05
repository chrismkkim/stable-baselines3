from typing import Callable, Optional

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
from matplotlib import pyplot as plt

from stable_baselines3.common.monitor import load_results

X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100


def rolling_window(array: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a rolling window to a np.ndarray

    :param array: the input Array
    :param window: length of the rolling window
    :return: rolling window on the input array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = (*array.strides, array.strides[-1])
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def window_func(var_1: np.ndarray, var_2: np.ndarray, window: int, func: Callable) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a function to the rolling window of 2 arrays

    :param var_1: variable 1
    :param var_2: variable 2
    :param window: length of the rolling window
    :param func: function to apply on the rolling window on variable 2 (such as np.mean)
    :return:  the rolling output with applied function
    """
    var_2_window = rolling_window(var_2, window)
    function_on_var2 = func(var_2_window, axis=-1)
    return var_1[window - 1 :], function_on_var2


def ts2xy(data_frame: pd.DataFrame, x_axis: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose a data frame variable to x and ys

    :param data_frame: the input data
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :return: the x and y output
    """
    if x_axis == X_TIMESTEPS:
        x_var = np.cumsum(data_frame.l.values)  # type: ignore[arg-type]
        y_var = data_frame.r.values
    elif x_axis == X_EPISODES:
        x_var = np.arange(len(data_frame))
        y_var = data_frame.r.values
    elif x_axis == X_WALLTIME:
        # Convert to hours
        x_var = data_frame.t.values / 3600.0  # type: ignore[operator, assignment]
        y_var = data_frame.r.values
    else:
        raise NotImplementedError
    return x_var, y_var  # type: ignore[return-value]


def plot_curves(
    xy_list: list[tuple[np.ndarray, np.ndarray]], x_axis: str, title: str, frac:str="all", figsize: tuple[int, int] = (8, 2)
) -> None:
    """
    plot the curves

    :param xy_list: the x and y coordinates to plot
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    """

    plt.figure(title, figsize=figsize)
    # max_x = max(xy[0][-1] for xy in xy_list)
    # min_x = 0
    for _, (x, y) in enumerate(xy_list):
        if frac == "all":
            None
        elif frac == "half":
            idx_half = int(len(x)/2)
            x = x[idx_half:] - x[idx_half]
            y = y[idx_half:]
        plt.scatter(x, y, s=4)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean)
    # plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()


def plot_results(
    dirs: list[str], num_timesteps: Optional[int], x_axis: str, task_name: str, frac: str="all", figsize: tuple[int, int] = (8, 2)
) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    plot_curves(xy_list, x_axis, task_name, frac, figsize)


def collect_topk(
    dirs: list[str], nagents: int, topk : int, num_timesteps: Optional[int], x_axis: str, frac: str="all", halftime : float=4e5
) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param figsize: Size of the figure (width, height)
    """

    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]

    # append all agents
    ycum = []
    for _, (x, y) in enumerate(xy_list):
        if frac == "all":
            None
        elif frac == "half":
            idx_half = x > halftime
            y = y[idx_half]
        ycum.append(y)
        
    # pick top k performers
    idx_k = pick_topk(nagents, ycum, topk)
    
    # min len of top k performers
    minlen = min_len(nagents, ycum, idx_k)
    
    # reward trajectory of top k performers
    yarr = np.zeros((topk,minlen))    
    for j, i in enumerate(idx_k):
        yarr[j] = ycum[i][:minlen]
    
    return yarr

def pick_topk(nagents, ycum, topk):
    # pick top k performers
    rewarr = np.zeros(nagents)
    for i in range(nagents):
        rewarr[i] = np.sum(ycum[i])
    idx_k  = np.argsort(rewarr)[-topk:]
    return idx_k    

def min_len(nagents, ycum, idx_k):
    lenarr = np.zeros(nagents).astype(int)
    for i in range(nagents):
        lenarr[i] = ycum[i].shape[0]
    minlen = np.min(lenarr[idx_k])
    return minlen
    

def plot_curves(
    xy_list: list[tuple[np.ndarray, np.ndarray]], x_axis: str, title: str, frac:str="all", figsize: tuple[int, int] = (8, 2)
) -> None:
    """
    plot the curves

    :param xy_list: the x and y coordinates to plot
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: the title of the plot
    :param figsize: Size of the figure (width, height)
    """

    plt.figure(title, figsize=figsize)
    # max_x = max(xy[0][-1] for xy in xy_list)
    # min_x = 0
    for _, (x, y) in enumerate(xy_list):
        if frac == "all":
            None
        elif frac == "half":
            idx_half = int(len(x)/2)
            x = x[idx_half:] - x[idx_half]
            y = y[idx_half:]
        plt.scatter(x, y, s=4)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean)
    # plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()
