from typing import Tuple
import numpy as np


def generate_points(
    centers: np.ndarray | list, sigmas: np.ndarray | list, random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data for n clusters centered on centers
    from a normal distribution with a variance of variance
    """
    np.random.seed(random_seed)  # Set seed for reproducibility
    xpts = np.zeros(1)
    ypts = np.zeros(1)
    labels = np.zeros(1)
    for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
        xpts = np.hstack((xpts, np.random.standard_normal(200) * xsigma + xmu))
        ypts = np.hstack((ypts, np.random.standard_normal(200) * ysigma + ymu))
        labels = np.hstack((labels, np.ones(200) * i))

    x_data = np.vstack((xpts, ypts))

    return x_data, labels


# https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html
def dataset_skfuzzy() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data for three clusters centered on (4,2), (1,6) and (5,6)
    from a normal distribution with a variance of (0.8,0.3), (0.5,0.5) and (1.1,0.7)
    """
    centers = [[4, 2], [1, 7], [5, 6]]
    sigmas = [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]]
    return generate_points(centers, sigmas)


def dataset_compact() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data for three clusters centered on (4,2), (1,6) and (5,6)
    from a normal distribution with a variance of (0.3,0.3), (0.3,0.3) and (0.3,0.3)
    """
    centers = [[4, 2], [1, 7], [5, 6]]
    sigmas = [[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]
    return generate_points(centers, sigmas)


def dataset_overlap() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates data for three clusters centered on (4,2), (1,6) and (5,6)
    from a normal distribution with a variance of (1.5,1.5), (1.5,1.5) and (1.5,1.5)
    """
    centers = [[4, 2], [1, 7], [5, 6]]
    sigmas = [[1.5, 1.5], [1.5, 1.5], [1.5, 1.5]]
    return generate_points(centers, sigmas)
