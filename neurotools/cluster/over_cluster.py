"""
Functions to help the cluster process !!!!
"""

import numpy as np
import pandas as pd
import scanpy as sc
from numba.typed import List
from scanpy import AnnData

from sklearn.neighbors import DistanceMetric

from ..utils.general import summarise_data_fast
from .cluster_helpers import n_different

import scipy.spatial as spatial

import warnings
warnings.filterwarnings("ignore")

dist = DistanceMetric.get_metric('manhattan')

def average(expr, labels, label_set):
    """Averages the expression by label.
    """
    label_indices = List()
    [label_indices.append(np.where(labels == label)[0]) for label in label_set]
    if type(expr) == pd.DataFrame:
        expr = expr.values
    avg_data = summarise_data_fast(expr, label_indices)

    return avg_data

def cluster_and_average(data: AnnData, expr: np.ndarray, resolution: float):
    """ Clusters the data using leiden cluster at a specified resolution &
        return the average expression of each gene in each cluster.
    """
    sc.tl.leiden(data, resolution=resolution)

    ##### For each one of the clusters, let's get the most similar cluster.
    labels = data.obs['leiden'].values.astype(str)
    label_set = np.unique(labels)
    label_indices = List()
    [label_indices.append(np.where(labels == label)[0]) for label in label_set]
    avg_data = summarise_data_fast(expr.values, label_indices)

    return avg_data, label_set

def get_pairs(data: AnnData, expr: np.ndarray, resolution: float):
    """ Clusters the data using leiden cluster at a specified resolution &
        return the average expression of each gene in each cluster.

        Also determines nearest neighbour of each average cluster, and orders
        the cluster comparisons to make by those with the greatest distance
        (in order to improve speed by comparing the most dissimiliar nearest
        neighbours first!)
    """
    sc.tl.leiden(data, resolution=resolution)

    ##### For each one of the clusters, let's get the most similar cluster.
    labels = data.obs['leiden'].values.astype(str)
    label_set = np.unique(labels)
    label_indices = List()
    [label_indices.append(np.where(labels == label)[0]) for label in label_set]
    avg_data = summarise_data_fast(expr.values, label_indices)

    #### Getting distances..
    # global dist
    # dist_mat = dist.pairwise(avg_data)
    # pairs = []
    # dists = []
    # for i, labeli in enumerate(label_set):
    #     min_dist = min(dist_mat[i,dist_mat[i,:]>0])
    #     min_index = np.where(dist_mat[i,:]==min_dist)[0][0]
    #     pairs.append( [labeli, label_set[min_index]] )
    #     dists.append( min_dist )
    # pairs = np.array(pairs, dtype='object')
    # dists = np.array(dists)
    # order = np.argsort(-dists)

    ### Getting pairs based on approximate neighbour search for speed...
    pairs = []
    dists = []
    point_tree = spatial.cKDTree(avg_data)
    for i, labeli in enumerate(label_set):
        nearest_info = point_tree.query(avg_data[i, :], k=2)
        nearest_dist = nearest_info[0][-1]
        nearest_index = nearest_info[1][-1]
        dists.append(nearest_dist)
        pairs.append( [labeli, label_set[nearest_index]] )
    pairs = np.array(pairs, dtype='object')
    dists = np.array(dists)
    order = np.argsort(-dists)
    print(dists[order])

    return pairs[order]

######### Version 3
### Main difference is that considers both up- and down- regulated genes!
def is_over_clustered(data: AnnData, pairs: np.array, max_de: int=2,
                         padj_cutoff: float=.01, logfc_cutoff: float=1):
    """ Checks if the data has been overclustered; i.e. no DE genes between a
        given cluster & it's most similar cluster.
    """
    over_clustered = True
    for pair in pairs:
        ##### Getting nearest neighbour cluster
        n_de = n_different(data, 'leiden', pair, logfc_cutoff, padj_cutoff)

        if n_de > max_de:
            over_clustered = False
            break

    return over_clustered, n_de

def over_cluster(data, expr, res: float=1, max_de: int=2, padj_cutoff: float=.01,
                 logfc_cutoff: float=2, max_iter: int=20, step_size: float=2,
                 verbose: bool=True):
    """Overclusters that data; defined when no DE genes between any given cluster
        and it's nearest neighbour is reached.
    """
    #avg_data, label_set = cluster_and_average_v2(data, expr, res)
    pairs = get_pairs(data, expr, res)
    over_clustered, n_de = is_over_clustered(data, pairs=pairs,
                                             #avg_data, #label_set,
                                                    max_de=max_de,
                                                    padj_cutoff=padj_cutoff,
                                                    logfc_cutoff=logfc_cutoff)

    #### Repeat this until over_clustered or max_iter reached...
    for i in range(max_iter):
        #### Check if overclustered from previous iteration
        if over_clustered:
            if verbose:
                print(f"Over-clustered at iteration {i} resolution {res} with "
                      f"{len(pairs)} clusters.")
            break
        elif verbose:
            print(f"Not over-clustered at iteration {i} resolution {res} with "
                  f"{len(pairs)} clusters; "
                  f"detected {n_de} de genes in a cluster and it's nearest neighbour.")

        res += step_size
        pairs = get_pairs(data, expr, res)
        over_clustered, n_de = is_over_clustered(data, pairs=pairs,
                                                 # avg_data, #label_set,
                                                 max_de=max_de,
                                                 padj_cutoff=padj_cutoff,
                                                 logfc_cutoff=logfc_cutoff)

    if not over_clustered and verbose:
        print(
            f"Warning, max iteration ({max_iter}) reached and not over-clustered "
            f"at resolution {res} with {len(pairs)} clusters; "
            f"detected {n_de} de genes in a cluster and it's nearest neighbour.")

######### Version 2
def is_over_clustered_v2(data: AnnData,
                         avg_data: np.ndarray, label_set: np.array,
                         max_de: int=2,
                         padj_cutoff: float=.01,
                      #score_cutoff: float=1
                      ):
    """ Checks if the data has been overclustered; i.e. no DE genes between a
        given cluster & it's most similar cluster.
    """
    over_clustered = True
    point_tree = spatial.cKDTree(avg_data)
    for i, label in enumerate(label_set):
        ##### Getting nearest neighbour cluster
        nearest_index = point_tree.query(avg_data[i, :], k=2)[1][-1]
        nearest_label = label_set[nearest_index]

        ##### Calling de genes
        sc.tl.rank_genes_groups(data, groupby='leiden', use_raw=False,
                                groups=[label], reference=nearest_label,
                                )

        #### Getting results
        # scores_rank = pd.DataFrame(data.uns['rank_genes_groups']['scores']
        #                                                 ).values[:, 0].astype(float)
        # # By taking abs, doesn't matter if de up in one cluster or the other.
        # scores_ranked = np.abs(scores_rank)
        padjs_rank = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj']
                                  ).values[:, 0].astype(float)

        # sig_bool = np.logical_and(scores_ranked > score_cutoff,
        #                           padjs_rank < padj_cutoff)
        # n_de = len(np.where(sig_bool)[0])
        n_de = len(np.where(padjs_rank < padj_cutoff)[0])

        if n_de > max_de:
            over_clustered = False
            break

    return over_clustered, n_de

def over_cluster_v2(data, expr, res: float=1, max_de: int=2, padj_cutoff: float=.01,
                 #score_cutoff: float=2,
                 max_iter: int=20, step_size: float=2,
                 verbose: bool=True):
    """Overclusters that data; defined when no DE genes between any given cluster
        and it's nearest neighbour is reached.
    """
    avg_data, label_set = cluster_and_average(data, expr, res)
    over_clustered, n_de = is_over_clustered(data, avg_data,
                                                    label_set,
                                                    max_de=max_de,
                                                    padj_cutoff=padj_cutoff,
                                                    #score_cutoff=score_cutoff
                                             )

    #### Repeat this until over_clustered or max_iter reached...
    for i in range(max_iter):
        #### Check if overclustered from previous iteration
        if over_clustered:
            if verbose:
                print(f"Over-clustered at iteration {i} resolution {res} with "
                      f"{len(label_set)} clusters.")
            break
        elif verbose:
            print(f"Not over-clustered at iteration {i} resolution {res} with "
                  f"{len(label_set)} clusters; "
                  f"detected {n_de} de genes in a cluster and it's nearest neighbour.")

        res += step_size
        avg_data, label_set = cluster_and_average(data, expr, res)
        over_clustered, n_de = is_over_clustered(data, avg_data,
                                                        label_set,
                                                        max_de=max_de,
                                                        padj_cutoff=padj_cutoff,
                                                      #score_cutoff=score_cutoff
                                                 )

    if not over_clustered and verbose:
        print(
            f"Warning, max iteration ({max_iter}) reached and not over-clustered "
            f"at resolution {res} with {len(label_set)} clusters; "
            f"detected {n_de} de genes in a cluster and it's nearest neighbour.")

######### Version 1
def is_over_clustered_v1(data: AnnData,
                         avg_data: np.ndarray, label_set: np.array,
                         max_de: int=2,
                         padj_cutoff: float=.01, logfc_cutoff: float=1):
    """ Checks if the data has been overclustered; i.e. no DE genes between a
        given cluster & it's most similar cluster.
    """
    over_clustered = True
    point_tree = spatial.cKDTree(avg_data)
    for i, label in enumerate(label_set):
        ##### Getting nearest neighbour cluster
        nearest_index = point_tree.query(avg_data[i, :], k=2)[1][-1]
        nearest_label = label_set[nearest_index]

        ##### Calling de genes
        sc.tl.rank_genes_groups(data, groupby='leiden', use_raw=False,
                                groups=[label], reference=nearest_label,
                                )

        #### Getting results
        logfcs_rank = pd.DataFrame(
            data.uns['rank_genes_groups']['logfoldchanges']
            ).values[:, 0].astype(float)
        padjs_rank = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj']
                                  ).values[:, 0].astype(float)

        sig_bool = np.logical_and(logfcs_rank > logfc_cutoff,
                                  padjs_rank < padj_cutoff)
        n_de = len(np.where(sig_bool)[0])

        if n_de > max_de:
            over_clustered = False
            break

    return over_clustered, n_de

def over_cluster_v1(data, expr, res: float=1, max_de: int=2, padj_cutoff: float=.01,
                 logfc_cutoff: float=2, max_iter: int=20, step_size: float=2,
                 verbose: bool=True):
    """Overclusters that data; defined when no DE genes between any given cluster
        and it's nearest neighbour is reached.
    """
    avg_data, label_set = cluster_and_average(data, expr, res)
    over_clustered, n_de = is_over_clustered_v1(data, avg_data,
                                                    label_set,
                                                    max_de=max_de,
                                                    padj_cutoff=padj_cutoff,
                                                    logfc_cutoff=logfc_cutoff)

    #### Repeat this until over_clustered or max_iter reached...
    for i in range(max_iter):
        #### Check if overclustered from previous iteration
        if over_clustered:
            if verbose:
                print(f"Over-clustered at iteration {i} resolution {res} with "
                      f"{len(label_set)} clusters.")
            break
        elif verbose:
            print(f"Not over-clustered at iteration {i} resolution {res} with "
                  f"{len(label_set)} clusters; "
                  f"detected {n_de} de genes in a cluster and it's nearest neighbour.")

        res += step_size
        avg_data, label_set = cluster_and_average(data, expr, res)
        over_clustered, n_de = is_over_clustered_v1(data, avg_data,
                                                        label_set,
                                                        max_de=max_de,
                                                        padj_cutoff=padj_cutoff,
                                                      logfc_cutoff=logfc_cutoff)

    if not over_clustered and verbose:
        print(
            f"Warning, max iteration ({max_iter}) reached and not over-clustered "
            f"at resolution {res} with {len(label_set)} clusters; "
            f"detected {n_de} de genes in a cluster and it's nearest neighbour.")
