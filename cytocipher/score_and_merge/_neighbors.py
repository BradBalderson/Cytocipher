"""
Functions for determining which clusters to compare.
"""

import numpy as np
import pandas as pd
import scanpy as sc

import scipy.spatial as spatial

from numba import jit, njit, prange
from numba.typed import List
from ..utils.general import summarise_data_fast

def general_neighbours(data: sc.AnnData,
                       labels: np.array, label_set: np.array,
                       #neigh_key: str, #Removing since will be depracated.
                       mnn_frac_cutoff: float):
    """Relies on scanpy run beforehand to determine the neighbourhood graph
        between individual observations. Then gets the cluster neighbours by
        determining the number of cells belonging to each cluster which are MNNs
        and using a proportion of MNNs as a cutoff on which clusters to compare.
    """
    #knn_adj_matrix = data.uns[neigh_key]['connectivities'].toarray() > 0
    knn_adj_matrix = data.obsp['connectivities'].toarray() > 0

    neighbours, dists, clust_dists = get_neighs_FAST(labels, label_set,
                                                     knn_adj_matrix,
                                                     mnn_frac_cutoff)
    return list(neighbours), list(dists), \
           pd.DataFrame(clust_dists, index=label_set, columns=label_set)

@jit(parallel=True, nopython=False)
def get_neighs_FAST(labels: np.array, label_set: np.array,
                    knn_adj_matrix: np.ndarray,
                    mnn_frac_cutoff: float):
    """ Get's the neighbourhoods using method described in doc-string
        of general_neighbours, VERY quickly.
    """

    ### Counting the MNNs for each cluster ###
    clust_dists = np.zeros((len(label_set), len(label_set)), dtype=np.float64)
    for i in prange( len(label_set) ):
        labeli = label_set[i]

        labeli_bool = labels == labeli

        labeli_knns = knn_adj_matrix[labeli_bool, :]

        for labelj in label_set[(i + 1):]:
            j = np.where(label_set == labelj)[0][0]

            labelj_bool = labels == labelj

            labelj_knns = knn_adj_matrix[labelj_bool,]

            nn_ij = labeli_knns[:, labelj_bool]
            nn_ji = labelj_knns[:, labeli_bool].transpose()
            mnn_bool = np.logical_and(nn_ij, nn_ji)

            n_total = np.logical_and(labeli_bool, labelj_bool).sum()
            mnn_dist = mnn_bool.sum() / n_total

            clust_dists[i, j] = mnn_dist
            clust_dists[j, i] = mnn_dist

    ##### Now converting this into neighbourhood information....
    neighbours = List()
    dists = List()
    for i, label in enumerate(label_set):
        neigh_bool = clust_dists[i, :] > mnn_frac_cutoff
        neighbours.append( label_set[neigh_bool] )
        dists.append( clust_dists[i, neigh_bool] )

    return neighbours, dists, clust_dists

################################################################################
   # The below are old cluster neighbourhood determining functions
   # This has been re-implemented in a general way above, allowing for
   # usage of scanpy to determine the neighbourhood graph, and then
   # getting the cluster neighbours by determine the number of cells belonging
   # to each cluster which are MNNs and using a proportion of MNN as a cutoff
   # on which clusters to compare.

################################################################################
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
