"""
Functions for determining which clusters to compare.
"""

import numpy as np
import pandas as pd
import scanpy as sc

import scipy.spatial as spatial

from numba import jit, njit, prange
from numba.typed import List
from ..utils.general import summarise_data_fast, get_true_indices, get_indices

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

    #### Calculating the proportion of MNNs between clusters
    label_set_dict = {label: i for i, label in enumerate(label_set)}
    labels_as_indices = np.array([label_set_dict[label_] for label_ in labels],
                                                                 dtype=np.int64)
    label_lens = np.array(
                       [len(np.where(labels==label)[0]) for label in label_set],
                          dtype=np.int64)
    clust_dists = get_clust_dists(labels_as_indices, len(label_set),
                                  label_lens,
                                  knn_adj_matrix, mnn_frac_cutoff)

    #### Calculating the neighbours that are closer than mnn_frac_cutoff
    neighbours, dists = get_neighs_FAST(label_set, mnn_frac_cutoff,
                                        clust_dists)

    return neighbours, dists, \
           pd.DataFrame(clust_dists, index=label_set, columns=label_set)

@njit(parallel=False)
def get_clust_dists(labels_as_indices: np.array, n_labels: int,
                    label_lens: np.array,
                    knn_adj_matrix: np.ndarray, mnn_frac_cutoff: float):
    """ Gets the distance between the clusters as the proportion of MNNs.
    """
    clust_dists = np.zeros((n_labels, n_labels), dtype=np.float64)

    #### Getting the cells which are MNNs
    knn_indices = np.where( knn_adj_matrix )
    for pairi in range( len(knn_indices[0]) ):
        i, j = knn_indices[0][pairi], knn_indices[1][pairi]

        labeli, labelj = labels_as_indices[i], labels_as_indices[j]
        clust_dists[labeli, labelj] += 1
        clust_dists[labelj, labeli] += 1

    #### Getting totals to get proportions...
    mnn_indices = np.where( clust_dists > 0 )
    for mnni in prange( len(mnn_indices[0]) ):
        i, j = mnn_indices[0][mnni], mnn_indices[1][mnni]
        if i==j:
            clust_dists[i, j] = 0
            continue

        total = label_lens[i] + label_lens[j]

        clust_dists[i,j] = clust_dists[i,j] / total

    return clust_dists

#@jit(parallel=True, forceobj=True, nopython=False)
#@njit #(parallel=True)
@jit(parallel=False, forceobj=True, nopython=False)
def get_neighs_FAST(label_set: np.array, mnn_frac_cutoff: float,
                    clust_dists: np.ndarray
                    ):
    """ Get's the neighbourhoods using method described in doc-string
        of general_neighbours, VERY quickly.
    """

    neigh_bools = clust_dists > mnn_frac_cutoff

    ##### Now converting this into neighbourhood information....
    neighbours = []
    dists = []
    for i, label in enumerate(label_set):
        #neigh_bool = clust_dists[i, :] > mnn_frac_cutoff
        #neigh_indices = get_true_indices( neigh_bool )
        neigh_indices = neigh_bools[i, :] #np.where(neigh_bool)[0]

        neighbours.append( label_set[neigh_indices] )
        dists.append( clust_dists[i,:][neigh_indices] )

    return neighbours, dists

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

@njit(parallel=False)
def get_clust_dists_OLD(labels: np.array, label_set: np.array,
                    knn_adj_matrix: np.ndarray, mnn_frac_cutoff: float):
    """ Gets the distance between the clusters as the proportion of MNNs.
    """
    clust_dists = np.zeros((len(label_set), len(label_set)), dtype=np.float64)
    for i in prange(len(label_set)):
        labeli = label_set[i]

        #labeli_indices = get_indices(labels, labeli)
        labeli_indices = np.where(labels == labeli)[0]

        labeli_knns = knn_adj_matrix[labeli_indices, :]

        for j in range((i + 1), len(label_set)):
            labelj = label_set[j]

            #labelj_indices = get_indices(labels, labelj)
            labelj_indices = np.where(labels == labelj)[0]

            labelj_knns = knn_adj_matrix[labelj_indices, :]

            nn_ij = labeli_knns[:, labelj_indices]
            nn_ji = labelj_knns[:, labeli_indices].transpose()
            mnn_bool = np.logical_and(nn_ij, nn_ji)

            # n_total = np.logical_or(labeli_bool, labelj_bool).sum()
            n_total = len(labeli_indices) + len(labelj_indices)
            mnn_dist = mnn_bool.sum() / n_total

            clust_dists[i, j] = mnn_dist
            clust_dists[j, i] = mnn_dist

    return clust_dists