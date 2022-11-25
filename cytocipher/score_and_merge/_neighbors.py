"""
Functions for determining which clusters to compare.
"""

import numpy as np
import pandas as pd

import scipy.spatial as spatial

from numba.typed import List
from ..utils.general import summarise_data_fast

def average(expr: pd.DataFrame, labels: np.array, label_set: np.array):
    """Averages the expression by label.
    """
    label_indices = List()
    [label_indices.append(np.where(labels == label)[0]) for label in label_set]
    if type(expr) == pd.DataFrame:
        expr = expr.values
    avg_data = summarise_data_fast(expr, label_indices)

    return avg_data

def enrich_neighbours(enrich_scores: pd.DataFrame,
                     labels: np.array, label_set: np.array, knn: int,
                     verbose: bool=True):
    """ Determines the neighbours based on those with the most similar
        enrichment scores.
    """

    neighbours = []
    dists = []

    if verbose:
        print("Getting nearest neighbours by enrichment scores.")

    avg_data = average(enrich_scores, labels, label_set)
    point_tree = spatial.cKDTree(avg_data)
    for i, labeli in enumerate(label_set):
        nearest_info = point_tree.query(avg_data[i, :], k=knn + 1)
        nearest_indexes = nearest_info[1]
        dists_ = nearest_info[0]

        neighbours.append([label_set[index] for index in nearest_indexes
                           if label_set[index] != labeli])
        dists.append([dist for i_, dist in enumerate(dists_)
                      if label_set[nearest_indexes[i_]] != labeli])

    return neighbours, dists

def all_neighbours(label_set: np.array):
    """Considers all clusters as neighbours, thus perform pair-wise comparisons.
    """
    neighbours = []
    dists = []

    for label in label_set:
        neighbours.append(list(label_set[label_set != label]))
        dists.append([np.nan] * (len(label_set) - 1))

    return neighbours, dists