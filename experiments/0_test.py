from functools import partial
import sys
import os

sys.path.append(os.path.abspath("."))

import skfuzzy as fuzz  # type: ignore

import matplotlib.pyplot as plt  # type: ignore
import numpy as np


from FCM.FCM import FCM
from datasets.load_datasets import (
    dataset_compact,
    dataset_overlap,
    dataset_skfuzzy,
)
from measures.measures import (
    partition_coefficient,
    partition_entropy,
    V1_index,
    cluster_overlap_index,
    index_on_cluster_overlap_index,
    xie_beni_index,
)
from measures.aggregation_functions import (
    o_prod,
    o_min,
    o_geo_mean,
    o_ob,
    o_odiv,
    Oz,
    Op,
    O,
    prob_sum,
    max_n_group,
    dual_gm,
    gb_grouping,
    gdiv_grouping,
)


def visualize_data(X, labels):
    # Visualize the test data
    colors = ["b", "orange", "g", "r", "c", "m", "y", "k", "Brown", "ForestGreen"]
    fig0, ax0 = plt.subplots()
    for label in range(3):
        ax0.plot(
            X[0, :][labels == label], X[1, :][labels == label], ".", color=colors[label]
        )
    ax0.set_title("Test data: 200 points x3 clusters.")
    plt.show()


def cluster_numbers():
    X, y = dataset_skfuzzy()
    # X, y = dataset_compact()
    # X, y = dataset_overlap()
    visualize_data(X, y)
    # V, U = FCM(X,3)
    # print(V)
    # print("a")

    overlap_functions = [o_prod, o_min, o_geo_mean, o_ob, o_odiv]
    n_grouping_functions = [
        prob_sum,
        max_n_group,
        dual_gm,
        gb_grouping,
        gdiv_grouping,
        np.mean,
    ]
    overlap_indices = [Oz, Op]
    overlap_indices_names = ["Oz", "Op"]
    for of in overlap_functions:
        for ng in n_grouping_functions:
            new_oi = partial(O, Of=of, M=ng)
            overlap_indices.append(new_oi)
            overlap_indices_names.append(
                f"O({str(new_oi).split()[5]},{str(new_oi).split()[9]})"
            )

    fig1, axes1 = plt.subplots(3, 3, figsize=(10, 10))
    alldata = (
        X / 10
    )  # /10 becasue overlaps work in 0-1, not in R #np.vstack((xpts, ypts))
    fpcs = []
    all_results = []
    for ncenters, ax in enumerate(axes1.reshape(-1), 2):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, ncenters, 2, error=0.005, maxiter=1000, init=None
        )

        results = []
        fpca = partition_coefficient(u)
        fpe = partition_entropy(u)
        xb = xie_beni_index(u, cntr, alldata.T)
        po = V1_index(u, cntr, alldata.T)
        results.append(fpca)
        results.append(fpe)
        results.append(xb)
        results.append(po)
        results_names = ["PC", "PE", "XB", "PO(Oz,AM,AM)"]
        for M1 in n_grouping_functions:
            for Oi, Oi_name in zip(overlap_indices, overlap_indices_names):
                poi = V1_index(u, cntr, alldata.T, m=2, O=Oi, M=M1)
                results.append(poi)
                results_names.append(f"M:{str(M1).split()[1]},PO:{Oi_name}")
        poc = index_on_cluster_overlap_index(
            u, cntr, alldata.T, np.mean, Oz, np.mean, np.mean, 0.3, 0.7
        )
        print(f"{ncenters=}, {fpc=}, {fpca=}, {fpe=}, {xb=}, {po=}, {poc=}")

        # Store fpc values for later
        fpcs.append(fpc)

        # Plot assigned clusters, for each data point in training set
        colors = ["b", "orange", "g", "r", "c", "m", "y", "k", "Brown", "ForestGreen"]
        cluster_membership = np.argmax(u, axis=0)
        # visualize_data(X, cluster_membership)  # But this creates a new plot instead of embeding in the subplots axes1
        for j in range(ncenters):
            ax.plot(
                X[0, :][cluster_membership == j],
                X[1, :][cluster_membership == j],
                ".",
                color=colors[j],
            )

        # Mark the center of each fuzzy cluster
        for pt in cntr * 10:
            ax.plot(pt[0], pt[1], "rs")

        # ax.set_title('Centers = {0}; FPC = {1:.2f};'.format(ncenters, fpc))
        # ax.set_title('FPC = {0:.2f}; PO = {1:.2f}; POC = {2:.2f};'.format(fpc, po, poc))
        ax.set_title("FPC = {0:.2f}; PO = {1:.2f};".format(fpc, po))
        ax.axis("off")

        all_results.append(results)

    fig1.tight_layout()
    plt.show()
    data = np.column_stack((np.array(results_names), np.array(all_results).T))


def test_datasets():
    fig1, axes1 = plt.subplots(3, 1, figsize=(5, 7))
    fpcs = []

    for idx, ds in enumerate([dataset_compact, dataset_skfuzzy, dataset_overlap]):
        X, y = ds()
        ncenters = 3
        alldata = (
            X / 10
        )  # /10 becasue overlaps work in 0-1, not in R #np.vstack((xpts, ypts))
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            alldata, ncenters, 2, error=0.005, maxiter=1000, init=None
        )

        fpca = partition_coefficient(u)
        fpe = partition_entropy(u)
        xb = xie_beni_index(u, cntr, alldata.T)
        po = V1_index(u, cntr, alldata.T)
        poc = index_on_cluster_overlap_index(
            u, cntr, alldata.T, np.mean, Oz, np.mean, np.mean, 0.3, 0.7
        )
        print(f"{ncenters=}, {fpc=}, {fpca=}, {fpe=}, {xb=}, {po=}, {poc=}")

        # Store fpc values for later
        fpcs.append(fpc)

        # Plot assigned clusters, for each data point in training set
        colors = ["b", "orange", "g", "r", "c", "m", "y", "k", "Brown", "ForestGreen"]
        cluster_membership = np.argmax(u, axis=0)
        # visualize_data(X, cluster_membership)  # But this creates a new plot instead of embeding in the subplots axes1
        ax = axes1[idx]
        for j in range(ncenters):
            ax.plot(
                X[0, :][cluster_membership == j],
                X[1, :][cluster_membership == j],
                ".",
                color=colors[j],
            )

        # Mark the center of each fuzzy cluster
        for pt in cntr * 10:
            ax.plot(pt[0], pt[1], "rs")

        # ax.set_title('Centers = {0}; FPC = {1:.2f};'.format(ncenters, fpc))
        # ax.set_title('FPC = {0:.2f}; PO = {1:.2f}; POC = {2:.2f};'.format(fpc, po, poc))
        ax.set_title("FPC = {0:.2f}; PO = {1:.2f};".format(fpc, po))
        ax.axis("off")

    fig1.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_datasets()
    cluster_numbers()
