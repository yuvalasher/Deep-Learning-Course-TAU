from typing import Tuple
import numpy as np
from sklearn.datasets import make_moons, make_circles
from consts import MOONS_NOISE, NOISE, RANDOM_NUM
import torch


def _activate_random_seed(random_seed: int) -> None:
    """
    Activating the randomness of torch & numpy by defined random seed
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def generate_X_y(size: int, random_num: int = RANDOM_NUM) -> Tuple[np.array, np.array]:
    """
    Generating data (X, y) for a classification task of moons by a random number and size of the points received.
    """
    _activate_random_seed(random_seed=random_num)
    x, y = make_moons(size, noise=MOONS_NOISE, random_state=random_num)
    return x, y


def generate_regression_data(lower_bound: int = -5, upper_bound: int = 5, num_points: int = 30,
                             random_num: int = RANDOM_NUM) -> Tuple[np.array, np.array]:
    """
    Generate X, y(label) for the regression task
    The data is defined from a num_points between the lower_bound to upper_bound and trigonometric function with little
    randomness
    """
    _activate_random_seed(random_seed=random_num)
    x = np.linspace(lower_bound, upper_bound, num_points)
    y = np.linspace(lower_bound, upper_bound, num_points)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx) * np.cos(yy) + NOISE * np.random.rand(xx.shape[0], xx.shape[1])
    return np.append(xx.flatten(), yy.flatten()).reshape(2, -1).T, z.reshape(-1)
    # return (np.append(xx.flatten(), yy.flatten()).reshape(2, -1)).reshape(900,2), z.flatten()
