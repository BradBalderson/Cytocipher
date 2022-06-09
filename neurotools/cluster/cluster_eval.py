"""
Functions for running automated diagnostics on cluster quality.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

from sklearn.metrics import silhouette_samples

def cluster_silhouettes(data: AnnData, groupby: str, n_rands: int=5000,
                        metric: str='cosine', verbose: bool=True):
    """ Determines the silhouette scores per cluster.

    Parameters
    ----------
    data: AnnData
        The data object.
    groupby: str
        The column in data.obs to group that define the clusters.
    n_rands: int
        Number of random cells to sample to calculate silhouette scores.
    Returns
    -------
    data.uns[f'{groupby}_sils']
        For a random set of cells, contains the silhouette scores for each cell
        using the per cluster gene enrichment scores as the variables.
    """
    if not f'{groupby}_enrich_scores' in data.obsm:
        raise Exception("Need to run giotto_page_enrich first with same groupby.")
    if groupby not in data.obs:
        raise Exception(f"{groupby} not found in data.obs.")

    cell_scores = data.obsm[f'{groupby}_enrich_scores'].values
    cluster_labels = data.obs[groupby].values.astype(str)

    rand_indices = np.random.choice(list(range(cell_scores.shape[0])), n_rands)

    scores = silhouette_samples(cell_scores[rand_indices, :],
                                cluster_labels[rand_indices], metric=metric)

    data.uns[f'{groupby}_sils'] = [scores, cluster_labels[rand_indices],
                                   rand_indices]
    if verbose:
        print(f"Added data.uns[f'{groupby}_sils']")

##### Have literally already done this in cluster_score.py
"""
def cluster_enrich(data, groupby: str, group_genes: dict=None,
                  padj_cutoff: float=.05, logfc_cutoff: float=None,
                  verbose: bool=True):
    Scores the different clusters based on the enrichment of cluster Limma_DE
        genes. Must have performed sc.tl.rank_genes_group first!

    Parameters
    ----------
    data: AnnData
        The data object.
    groupby: str
        The column in data.obs to group the data by, must have performed
        sc.tl.rank_genes_groups first on this if group_genes not specified.
    group_genes: dict
        Dictionary mapping groups in groupby to genes in data.var_names.
    Returns
    -------
    data.obsm[f'{groupby}_scores']
        Columns are the enrichment of the cell for genes that are markers for
        each cluster.
        
    # Getting the cluster labels #
    if groupby in data.obs:
        cluster_labels = data.obs[groupby].values.astype(str)
    else:
        raise Exception(f"{groupby} not in data.obs.")

    # Getting the markers for each group #
    if type(group_genes) == type(False):
        genes_ranked = pd.DataFrame(data.uns['rank_genes_groups']['names'])
        padjs = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj'])
        sig = padjs.values < padj_cutoff

        if type(logfc_cutoff)!=type(None):
            logfcs = pd.DataFrame(data.uns['rank_genes_groups'][
                                                              'logfoldchanges'])
            sig = np.logical_and(sig, logfcs.values)

        group_genes = {}
        for i, group in enumerate(genes_ranked.columns):
            if group not in cluster_labels:
                raise Exception(
                    f"Appears sc.tl.rank_genes_groups run on different column, "
                    f"since found label {group} not present in "
                    f"data.obs['{groupby}']")
            group_genes[group] = genes_ranked.values[sig[:,i], i]

    # Precalculations..
    fcs, mean_fcs, std_fcs = calc_page_enrich_input(data)

    cell_scores = np.zeros((data.shape[0], len(group_genes)))
    #cluster_scores = np.zeros((len(group_genes), len(group_genes)))
    for i, clusteri in enumerate(group_genes):
        if len(group_genes[clusteri]) == 0: # No Limma_DE.
            if verbose:
                print(f"Warning, detected cluster with no Limma_DE ({clusteri}).")
            continue

        cluster_scores_ = giotto_page_enrich_min(group_genes[clusteri],
                                                    data.var_names, fcs,
                                                    mean_fcs, std_fcs)
        cell_scores[:, i] = cluster_scores_

        # for j, clusterj in enumerate(group_genes):
        #     cluster_scores[i, j] = np.mean(
        #                             cluster_scores_[cluster_labels == clusterj])

    data.obsm[f'{groupby}_scores'] = cell_scores
    data.uns[f'{groupby}_markers'] = group_genes
    if verbose:
        print(f"Added data.obsm[f'{groupby}_scores']")
        print(f"Added data.uns[f'{groupby}_markers']")
"""
