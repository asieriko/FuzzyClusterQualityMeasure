from collections.abc import Callable
from math import log
import numpy as np

from measures.aggregation_functions import Oz


def compactness(
    u: np.ndarray, centers: np.ndarray, points: np.ndarray, m: int = 2
) -> float:
    """
    Compactness measure
    :math:`\frac{1}{n} \sum_i \sum_j u_{ij}^m ||v_i - x_j||^2`.

    Parameters
    ----------
    u : np.array (centers x points)
        membership of each point to each center
    centers: np.array (centers x dimensions)
        coordinates of each  center of the clusters
    points: np.array (points x dimensions)
        coordinates of each point
    m: int
        fuzziness degree

    Returns
    -------
    float
        Compactness measure of the given partition respect to the centers
    """
    c, n = u.shape
    xb_compactness = 0.0
    for i in range(c):
        for j in range(n):
            xb_compactness += u[i, j] ** m * np.sqrt(sum((centers[i] - points[j]) ** 2))
    xb_compactness = xb_compactness / n
    return xb_compactness


def partition_coefficient(u: np.ndarray, m: int = 2) -> float:
    """
    Partition coefficient
    :math:`\frac{1}{n} \sum_i \sum_j u_{ij}^m`.

    J. C. Bezdek, Pattern Recognition with Fuzzy Objective Function Algorithms. Springer US, 1981.
    https://doi.org/10.1007/978-1-4757-0450-1

    Parameters
    ----------
    u : np.array (centers x points)
        membership of each point to each center
    m: int
        fuzziness degree

    Returns
    -------
    float
        Partition coefficient of the given partition
    """
    c, n = u.shape
    vpc = 0.0
    for i in range(c):
        for j in range(n):
            vpc += u[i, j] ** m
    vpc = vpc / n
    return vpc


def partition_entropy(u: np.ndarray, a: int = 2) -> float:
    """
    Partition entropy
    :math:`- \frac{1}{n} \sum_i \sum_j u_{ij}^m log_a(u_{ij})`.

    J. C. Bezdek, Pattern Recognition with Fuzzy Objective Function Algorithms. Springer US, 1981.
    https://doi.org/10.1007/978-1-4757-0450-1

    Parameters
    ----------
    u : np.array (centers x points)
        membership of each point to each center
    centers: np.array (centers x dimensions)
        coordinates of each  center of the clusters
    points: np.array (points x dimensions)
        coordinates of each point
    a: int


    Returns
    -------
    float
        Partition entropy of the given partition respect to the centers
    """
    c, n = u.shape
    vpe = 0.0
    for i in range(c):
        for j in range(n):
            vpe += u[i, j] * log(u[i, j], a)
    vpe = -vpe / n
    return vpe


def xie_beni_index(
    u: np.ndarray, centers: np.ndarray, points: np.ndarray, m: int = 2
) -> float:
    """
    Xie-Beni index
    :math:`compactness = \frac{1}{n} \sum_i \sum_j u_{ij}^m ||v_i - x_j||^2`.
    :math:`separation = min{||u_i - u_j||^2}; \forall i,j: i \neq j`.
    :math:`compactness / separation`.

    X. Xie and G. Beni (1991)
    https://doi.ieeecomputersociety.org/10.1109/34.85677

    Parameters
    ----------
    u : np.array (centers x points)
        membership of each point to each center
    centers: np.array (centers x dimensions)
        coordinates of each  center of the clusters
    points: np.array (points x dimensions)
        coordinates of each point
    m: int
        fuzziness degree

    Returns
    -------
    float
        Xie Beni index of the given partition respect to the centers
    """
    # c,n=u.shape/
    xb_compactness = compactness(u, centers, points, m=m)
    # xb_separation = min(centers)
    # https://stackoverflow.com/questions/63673658/compute-distances-between-all-points-in-array-efficiently-using-python
    xb_separation = min(
        np.abs(centers[np.newaxis, :, :] - centers[:, np.newaxis, :]).min(axis=2)[
            np.triu_indices(centers.shape[0], 1)
        ]
    )
    xb = xb_compactness / xb_separation
    print("XB", xb_compactness, xb_separation)
    return xb


def V1_index(
    u: np.ndarray,
    centers: np.ndarray,
    points: np.ndarray,
    m: int = 2,
    O: Callable = Oz,
    M: Callable = np.mean,
    F: Callable = np.mean,
) -> float:
    """
    Compactness measure
    :math:`compactness = \frac{1}{n} \sum_i \sum_j u_{ij}^m ||v_i - x_j||^2`.
    :math:`PO = M(\mathcal{O}_{i \neq j}(u_i,u_j)`.
    :math:`V1 = F(compactness / PO)`.

    X. Xie and G. Beni (1991)
    https://doi.ieeecomputersociety.org/10.1109/34.85677

    Parameters
    ----------
    u : np.array (centers x points)
        membership of each point to each center
    centers: np.array (centers x dimensions)
        coordinates of each  center of the clusters
    points: np.array (points x dimensions)
        coordinates of each point
    m: int
        fuzziness degree
    O: function
        Overlap index
    M: function
        arithmetic mean or a n-dimensional grouping function
    F: function
        bivaraite aggreagation function

    Returns
    -------
    float
        V1 index of the given partition respect to the centers
    """
    c, n = u.shape
    compactness_value = compactness(u, centers, points, m=m)

    partials = []
    for i in range(c):
        for j in range(i, c):
            if i == j:
                continue
            partials.append(O(u[i, :], u[j, :]))
    po = M(partials)

    v1 = F([compactness_value, po])
    # print("PO",compactness_value, po)
    return v1


def W(x, epsilon, delta):
    """
    x: float
    epsilon, sigma in [0,1]
    epsilon < sigma
    """
    assert 0 <= epsilon < delta <= 1
    if x < epsilon:
        return 1
    if x < delta:
        return 1 - x
    return 0


def cluster_overlap_index(u, centers, O, M, epsilon, delta):
    """
    O: Overlap index
    epsilon, sigma in [0,1]
    epsilon < sigma
    M: Aggregation function
    """
    oc = M(
        [O(u[0], u[1]), W(np.sqrt(sum((centers[0] - centers[1]) ** 2)), epsilon, delta)]
    )
    return oc


def index_on_cluster_overlap_index(
    u, centers, points, Ag, O, Moc, Mi, epsilon, delta, m=2
):
    """
    O: Overlap index
    epsilon, sigma in [0,1]
    epsilon < sigma
    M: Aggregation function
    """
    c, n = u.shape

    compactness_value = compactness(u, centers, points, m=m)

    ocs = []
    for i in range(c):
        for j in range(i, c):
            if i == j:
                continue
            ocs.append(
                cluster_overlap_index(
                    u[[i, j], :], centers[[i, j], :], O, Moc, epsilon, delta
                )
            )

    poc = Mi(ocs)
    idx = Ag([compactness_value, poc])
    print("POC", compactness_value, poc)
    return idx
