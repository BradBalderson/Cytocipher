"""
Functions for measuring quality of clusters.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

def calc_page_enrich_input(data):
    """ Calculates stats necessary to calculate enrichment score.
    """
    full_expr = data.to_df().values

    gene_means = full_expr.mean(axis=0)
    fcs = np.apply_along_axis(np.subtract, 1, full_expr, gene_means)
    mean_fcs = np.apply_along_axis(np.mean, 1, fcs)
    std_fcs = np.apply_along_axis(np.std, 1, fcs)

    return fcs, mean_fcs, std_fcs

def giotto_page_enrich_min(gene_set, var_names, fcs, mean_fcs, std_fcs):
    """ Calculates enrichment scores with most values pre-calculated.
    """
    gene_indices = [np.where(var_names == gene)[0][0] for gene in gene_set]
    set_fcs = np.apply_along_axis(np.mean, 1, fcs[:, gene_indices])

    giotto_scores = ((set_fcs - mean_fcs) * np.sqrt(len(gene_indices)))/std_fcs
    return giotto_scores

def giotto_page_enrich_geneset(data, gene_set):
    """ Re-implementation of giotto page-enrichment score.
    """
    fcs, mean_fcs, std_fcs = calc_page_enrich_input(data)
    return giotto_page_enrich_min(gene_set, data.var_names.values,
                                  fcs, mean_fcs, std_fcs)

def giotto_page_enrich(data: AnnData, groupby: str,
                       var_groups: str='highly_variable',
                       logfc_cutoff: float=0, padj_cutoff: float=.05,
                       n_top: int=5, cluster_marker_key: str=None,
                       rerun_de: bool=True,
                       verbose: bool=True):
    """ Runs Giotto coexpression enrichment score for DE genes in each cluster.
    """
    n_top = data.shape[1] if type(n_top)==type(None) else n_top

    #### First performing differential expression...
    if type(cluster_marker_key)==type(None):
        if type(var_groups)!=type(None):
            data_sub = data[:,data.var[var_groups]]
            sc.tl.rank_genes_groups(data_sub, groupby=groupby, use_raw=False)
            data.uns['rank_genes_groups'] = data_sub.uns['rank_genes_groups']
        elif rerun_de:
            sc.tl.rank_genes_groups(data, groupby=groupby, use_raw=False)

        #### Getting marker genes for each cluster...
        genes_rank = pd.DataFrame(data.uns['rank_genes_groups']['names'])
        logfcs_rank = pd.DataFrame(data.uns['rank_genes_groups']['logfoldchanges'])
        padjs_rank = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj'])

        up_bool = np.logical_and(logfcs_rank.values > logfc_cutoff,
                                 padjs_rank.values < padj_cutoff)

        cluster_genes = {}
        for i, cluster in enumerate(genes_rank.columns):
            up_indices = np.where(up_bool[:,i])[0]
            up_rank = np.argsort(-logfcs_rank.values[up_indices, i])[0:n_top]
            cluster_genes[cluster] = genes_rank.values[up_rank, i]

        data.uns[f'{groupby}_markers'] = cluster_genes
        cluster_marker_key = f'{groupby}_markers'
        if verbose:
            print(f"Added data.uns['{groupby}_markers']")

    cluster_genes = data.uns[cluster_marker_key]

    ###### Getting the enrichment scores...
    # Precalculations..
    fcs, mean_fcs, std_fcs = calc_page_enrich_input(data)

    cell_scores = np.zeros((data.shape[0], len(cluster_genes)))
    for i, clusteri in enumerate(cluster_genes):
        cluster_scores_ = giotto_page_enrich_min(cluster_genes[clusteri],
                                                    data.var_names, fcs,
                                                    mean_fcs, std_fcs)
        cell_scores[:, i] = cluster_scores_

    ###### Adding to AnnData
    cluster_scores = pd.DataFrame(cell_scores, index=data.obs_names,
                                  columns=list(cluster_genes.keys()))
    data.obsm[f'{groupby}_enrich_scores'] = cluster_scores

    if verbose:
        print(f"Added data.obsm['{groupby}_enrich_scores']")
