from typing import Callable
from math import sqrt
import numpy as np


# Overlap functions
def o_prod(x: float, y: float) -> float:
    """
    Product overlap function
    :math:`o\_prod = x \cdot y`.

    Parameters
    ----------
    x: float
    y: float

    Returns
    -------
    float
       product of x and y
    """
    return x * y


def o_min(x: float, y: float) -> float:
    """
    minimum overlap function
    :math:`o\_min = x \cdot y`.

    Parameters
    ----------
    x: float
    y: float

    Returns
    -------
    float
       minimum of x and y
    """
    return min(x, y)


def o_geo_mean(x: float, y: float) -> float:
    """
    geometric mean overlap function
    :math:`o\_geo\_mean = \sqrt{x \cdot y}`.

    Parameters
    ----------
    x: float
    y: float

    Returns
    -------
    float
       geometric mean of x times y
    """
    return sqrt(x * y)


def o_ob(x: float, y: float) -> float:
    """
    OB overlap function
    :math:`o\_OB = \sqrt{x \cdot y \cdot min(x,y)}`.

    Parameters
    ----------
    x: float
    y: float

    Returns
    -------
    float
       OB overlap of x and y
    """
    return sqrt(x * y * min(x, y))


def o_odiv(x: float, y: float) -> float:
    """
    ODOV overlap function
    :math:`o\_OB = \frac{x \cdot y + min(x,y)}{2}`.

    Parameters
    ----------
    x: float
    y: float

    Returns
    -------
    float
       ODIV overlap of x and y
    """
    return sqrt(x * y + min(x, y)) / 2


# Overlap indices
def Oz(A: np.ndarray, B: np.ndarray) -> float:
    """
    Overlap index - Zadeh's consistency index
    :math:`O_z = max min(A_i, B_i)`.

    Parameters
    ----------
    A: np.array
        Fuzzy set A
    B: np.array
        Fuzzy set B

    Returns
    -------
    float
        Overlap index of the given fuzzy sets
    """
    O = max(np.minimum(A, B))
    return O


def Op(A: np.ndarray, B: np.ndarray) -> float:
    """
    Overlap index such that
    :math:`O_p = \frac{1}{n} \sum(A_i \cdot B_i)`.

    Parameters
    ----------
    A: np.array
        Fuzzy set A
    B: np.array
        Fuzzy set B

    Returns
    -------
    float
        Overlap index of the given fuzzy sets
    """
    O = np.mean(A * B)
    return O


def O(A: np.ndarray, B: np.ndarray, Of: Callable, M: Callable) -> float:
    """
    Generic overlap index from a function M and an overlap function Of
    :math:`O = M(O(A_i, B_i))`.

    Parameters
    ----------
    A: np.array
        Fuzzy set A
    B: np.array
        Fuzzy set B
    Of: function
        Overlap function
    M: function
        arithmetic mean or a n-dimensiona grouping function

    Returns
    -------
    float
        Overlap index of the given fuzzy sets
    """
    Oi = []
    for ai, bi in zip(A, B):
        Oi.append(Of(ai, bi))
    return M(Oi)


# n-dimensional grouping functions
def prob_sum(x: np.ndarray) -> float:
    """
    Probabilistic sum
    :math:`prob\_sum = 1 - \prod{(1 - x_i)}`.

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        n-dimensional probabilistic sum of the given vector
    """
    ps = 1
    for xi in x:
        ps = ps * (1 - xi)
    return 1 - ps


def max_n_group(x: np.ndarray) -> float:
    """
    n-dimensional grouping: Maximum
    :math:`max = max(x_i)`.

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        n-dimensional maximum of the given vector
    """
    return np.max(x)


def dual_gm(x: np.ndarray) -> float:
    """
    Dual of Geometric mean
    :math:`dual\_gm = 1 - [\prod{(1 - x_i)}]^\frac{1}{n}`.

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        dual of the geometric mean of the given vector
    """
    n = len(x)
    ps = 1
    for xi in x:
        ps = ps * (1 - xi)
    return 1 - (ps ** (1 / n))


def gb_grouping(x: np.ndarray) -> float:
    """
    GB n-dimensional grouping function
    :math:`gb\_grouping = 1 - \sqrt{\prod{(1 - x_i)}\cdot min(1-x_1,\dots,1-x_n)}`.

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        dual of the geometric mean of the given vector
    """
    ps = []
    for xi in x:
        ps.append(1 - xi)
    return 1 - sqrt(np.prod(ps) * np.min(ps))


def gdiv_grouping(x: np.ndarray) -> float:
    """
    GDIV n-dimensional grouping function
    :math:`gdiv\_grouping = 1 - \frac{\prod{(1 - x_i)} + min(1-x_1,\dots,1-x_n)}{2}`.

    Parameters
    ----------
    x: np.array

    Returns
    -------
    float
        dual of the geometric mean of the given vector
    """
    ps = []
    for xi in x:
        ps.append(1 - xi)
    return 1 - (np.prod(ps) + np.min(ps)) / 2
