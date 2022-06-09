"""
Neurotools automated labelling & merging of clusters based on Limma_DE genes that were used to construct the
nearest neighbour graph for clustering & UMAP construction.

Works by labelling clusters based on Limma_DE genes (user defined criteria), & then
for any cluster with no Limma_DE genes based on this criteria, gets merged with the most
similar cluster to it. Then repeat attempt to label, and so on until all clusters
labelled with top Limma_DE genes from neurotools balanced feature selection.
"""

import numpy as np
import pandas as pd
import scanpy as sc

from numba.typed import List

import scipy.spatial as spatial
from ..utils.general import summarise_data_fast
from ..plotting.utils import add_colors

def average(expr: pd.DataFrame, labels: np.array, label_set: np.array):
    """Averages the expression by label.
    """
    label_indices = List()
    [label_indices.append(np.where(labels == label)[0]) for label in label_set]
    if type(expr) == pd.DataFrame:
        expr = expr.values
    avg_data = summarise_data_fast(expr, label_indices)

    return avg_data


def get_pairs(avg_data: np.ndarray, label_set: np.array):
    """Gets nearest neighbour to each datapoint.
    """
    pairs = []
    dists = []
    point_tree = spatial.cKDTree(avg_data)
    for i, labeli in enumerate(label_set):
        nearest_info = point_tree.query(avg_data[i, :], k=2)
        nearest_dist = nearest_info[0][-1]
        nearest_index = nearest_info[1][-1]
        dists.append(nearest_dist)
        pairs.append([labeli, label_set[nearest_index]])
    pairs = np.array(pairs, dtype='object')
    dists = np.array(dists)
    order = np.argsort(-dists)

    return pairs[order]

def get_genesubset_de(data: sc.AnnData, data_sub: sc.AnnData,
                      groupby: str, de_key: str):
    """ Gets Limma_DE genes for subset of the data, but adds to the full data.
    """
    sc.tl.rank_genes_groups(data_sub, groupby=groupby, key_added=de_key)
    data.uns[de_key] = data_sub.uns[de_key]

def label_clusters(data: sc.AnnData, groupby: str, de_key: str,
                   reference_genes: np.array,
                   max_genes: int, min_de: int, t_cutoff: float,
                   logfc_cutoff: float, padj_cutoff: float
                   ):
    """ Labels the clusters based on top Limma_DE genes.
    """
    #### Calling which genes are Limma_DE
    genes_df = pd.DataFrame(data.uns[de_key]['names'])
    tvals_df = pd.DataFrame(data.uns[de_key]['scores'])
    padjs_df = pd.DataFrame(data.uns[de_key]['pvals_adj'])
    logfcs_df = pd.DataFrame(data.uns[de_key]['logfoldchanges'])

    de_bool = tvals_df.values >= t_cutoff
    if padj_cutoff < 1 or logfc_cutoff > 0:  # necessary to add other criteria...
        de_bool = np.logical_and(de_bool, padjs_df.values < padj_cutoff)
        de_bool = np.logical_and(de_bool, logfcs_df.values >= logfc_cutoff)

    ##### Labelling clusters based on Limma_DE genes
    label_map = {}
    for i, cluster in enumerate(genes_df.columns):
        de_indices = np.where(de_bool[:, i])[0]
        if len(de_indices) >= min_de:  # Significant number of Limma_DE genes detected !!!!
            genes_ = genes_df.values[:, i]
            de_genes = np.unique(genes_[de_indices][0:max_genes])  # Sorts alphabetically

            # Need to put the reference genes first...
            if type(reference_genes) != type(None):
                ref_de = [gene for gene in de_genes if gene in reference_genes]
                other_de = [gene for gene in de_genes if
                            gene not in reference_genes]
                if len(ref_de) == 0:  # If no reference genes Limma_DE, then put the first reference gene with max t-value
                    ref_indices = [np.where(genes_ == ref)[0][0] for ref in
                                   reference_genes]
                    highest_index = np.argmax(tvals_df.values[ref_indices, i])
                    ref_de = [reference_genes[highest_index]]
                de_genes = ref_de + other_de

            label_map[cluster] = '-'.join(de_genes[0:max_genes])

        else:  # Non-significant number of Limma_DE genes, cluster isn't labelled.
            label_map[cluster] = cluster

    new_labels = np.array(
        [label_map[clust] for clust in data.obs[groupby].values])

    return label_map, new_labels

def merge_neighbours(expr: pd.DataFrame, cluster_labels: np.array, ):
    """ Merges unlabelled clusters to most similar neighbour.
    """
    ##### Getting the neighbours of each cluster based on average expression
    label_set = np.unique(cluster_labels)
    avg_data = average(expr, cluster_labels, label_set)

    label_pairs = get_pairs(avg_data, label_set)

    #### Getting groups of clusters which will be merged...
    merge_groups = []
    for pair in label_pairs:
        added = False
        if pair[0].isdigit():
            for merge_group in merge_groups:
                if np.any([pair_ in merge_group for pair_ in pair]):
                    merge_group.extend(pair)
                    added = True
                    break

            if not added:
                merge_groups.append(list(pair))

    #### Getting mapping from current clusters to merge clusters
    cluster_map = {}
    for i in range(len(merge_groups)):  # For the merge groups
        for cluster in merge_groups[i]:
            cluster_map[cluster] = str(i)

    for label in label_set:  # Simply map to the same cluster if not merging
        if label not in cluster_map:
            cluster_map[label] = label

    merge_cluster_labels = np.array(
        [cluster_map[clust] for clust in cluster_labels])

    return cluster_map, merge_cluster_labels

def add_labels(data: sc.AnnData, merge_col: str, cluster_labels: np.array,
               verbose: bool):
    """Adds labels to data"""
    data.obs[merge_col] = cluster_labels
    data.obs[merge_col] = data.obs[merge_col].astype('category')
    if verbose:
        print(f"Added data.obs[ '{merge_col}' ]")

def cluster_label(data: sc.AnnData, var_key: str,
                  groupby: str = 'leiden',
                  obs_key: str = 'cluster_labels',
                  # Stores final cluster labellings!
                  reference_genes: np.array = None,
                  # reference genes used for nts bfs, put these genes first for cluster label
                  max_genes: int = 5, min_de: int = 1, t_cutoff: float = 10,
                  de_key: str = 'rank_genes_groups',
                  logfc_cutoff: float = 0, padj_cutoff: float = 1,
                  verbose: bool = True,
                  ):
    """ Labels clusters by the top Limma_DE genes, and subsequently merges clusters
        with no Limma_DE genes to the most similar cluster.
    """

    if verbose:
        print("Start no. clusters: ",
              len(np.unique(data.obs[groupby].values)))

    #### Getting genes of interest
    genes_bool = data.var[var_key].values

    ### First calling differential expression
    data_sub = data[:, genes_bool]
    get_genesubset_de(data, data_sub, groupby, de_key)

    expr = data_sub.to_df()

    ### Now label clusters based on top Limma_DE genes according to criteria
    label_map, cluster_labels = label_clusters(data, groupby, de_key,
                                               reference_genes,
                                               max_genes, min_de, t_cutoff,
                                               logfc_cutoff, padj_cutoff)

    ### Now performing progressive merging if not all clusters labelled!
    # NOTE: each iteration has two mergings; merge based on labels & based on neighbour merging
    i = 0
    while np.any([label_map[key].isdigit() for key in
                  label_map]):  # Still clusters without label.
        i += 1

        #### Adding intermediate clustering results...
        merge_col = f'{groupby}_merge{i}'
        add_labels(data, merge_col, cluster_labels, verbose)

        #### Merging unlabelled clusters to most similar cluster
        label_map, cluster_labels = merge_neighbours(expr, cluster_labels)
        i += 1
        merge_col = f'{groupby}_merge{i}'
        add_labels(data, merge_col, cluster_labels, verbose)

        #### Calling de
        add_labels(data_sub, merge_col, cluster_labels, False)
        get_genesubset_de(data, data_sub, merge_col, de_key)

        #### Relabelling
        label_map, cluster_labels = label_clusters(data, merge_col,
                                                   # Now labelling based on merge_col
                                                   de_key, reference_genes,
                                                   max_genes, min_de, t_cutoff,
                                                   logfc_cutoff, padj_cutoff)

    #### Iteration complete, so can now add final labelled clusters.
    add_labels(data, obs_key, cluster_labels, verbose)
    add_colors(data, obs_key, 'tab20')  # Also adding colors
    if verbose:
        print("Final no. clusters: ", len(np.unique(cluster_labels)))

