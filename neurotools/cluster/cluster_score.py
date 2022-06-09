"""
Functions for measuring quality of clusters.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

from numba import njit

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
                       rerun_de: bool=True, gene_order='logfc',
                       verbose: bool=True):
    """ Runs Giotto coexpression enrichment score for Limma_DE genes in each cluster.
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
            if gene_order=='logfc':
                up_rank = np.argsort(-logfcs_rank.values[up_indices, i])[0:n_top]
            else:
                up_rank = up_indices[0:n_top]

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

################################################################################
             # Functions related to Coexpression Score #
################################################################################
@njit
def coexpr_score(expr: np.ndarray, min_counts: int = 2):
    """Enriches for the genes in the data"""

    expr_bool = expr > 0
    coexpr_counts = expr_bool.sum(axis=1)

    # min_counts = 2 ### Must be coexpression of atleast 2 of the markers!
    nonzero_indices = np.where(coexpr_counts > 0)[0]
    coexpr_indices = np.where(coexpr_counts >= min_counts)[0]
    cell_scores = np.zeros((expr.shape[0]), dtype=np.float64)
    for i in coexpr_indices:
        expr_probs = np.zeros((coexpr_counts[i]))
        cell_nonzero = np.where(expr_bool[i, :])[0]
        for j, genej in enumerate(cell_nonzero):
            expr_probs[j] = len(
                np.where(expr[nonzero_indices, genej] >= expr[i, genej])[0]) / \
                            expr.shape[0]

        cell_scores[i] = np.log2(coexpr_counts[i] / np.prod(expr_probs))

    return cell_scores


def get_markers(data: sc.AnnData, groupby: str,
                var_groups: str = 'highly_variable',
                logfc_cutoff: float = 0, padj_cutoff: float = .05,
                n_top: int = 5, rerun_de: bool = True, gene_order='logfc',
                verbose: bool = True):
    """Gets the marker genes as a dictionary...
    """
    if rerun_de:
        if type(var_groups) != type(None):
            data_sub = data[:, data.var[var_groups]]
            sc.tl.rank_genes_groups(data_sub, groupby=groupby, use_raw=False)
            data.uns['rank_genes_groups'] = data_sub.uns['rank_genes_groups']
        else:
            sc.tl.rank_genes_groups(data, groupby=groupby, use_raw=False)

    #### Getting marker genes for each cluster...
    genes_rank = pd.DataFrame(data.uns['rank_genes_groups']['names'])
    logfcs_rank = pd.DataFrame(data.uns['rank_genes_groups']['logfoldchanges'])
    padjs_rank = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj'])

    up_bool = np.logical_and(logfcs_rank.values > logfc_cutoff,
                             padjs_rank.values < padj_cutoff)

    cluster_genes = {}
    for i, cluster in enumerate(genes_rank.columns):
        up_indices = np.where(up_bool[:, i])[0]
        if gene_order == 'logfc':
            up_rank = np.argsort(-logfcs_rank.values[up_indices, i])[0:n_top]
        else:
            up_rank = up_indices[0:n_top]

        cluster_genes[cluster] = genes_rank.values[up_rank, i]

    data.uns[f'{groupby}_markers'] = cluster_genes
    cluster_marker_key = f'{groupby}_markers'
    if verbose:
        print(f"Added data.uns['{groupby}_markers']")


def coexpr_enrich(data: sc.AnnData, groupby: str,
                  cluster_marker_key: str = None, min_counts: int = 2,
                  verbose: bool = True):
    """ NOTE: unlike the giotto function version, this one assumes have already done DE.
    """
    if type(cluster_marker_key) == type(None):
        cluster_marker_key = f'{groupby}_markers'

    cluster_genes = data.uns[cluster_marker_key]

    ###### Getting the enrichment scores...
    cell_scores = np.zeros((data.shape[0], len(cluster_genes)))
    for i, clusteri in enumerate(cluster_genes):
        genes_ = cluster_genes[clusteri]
        cluster_scores_ = coexpr_score(data[:, genes_].X.toarray(),
                                       min_counts=min_counts)
        cell_scores[:, i] = cluster_scores_

    ###### Adding to AnnData
    cluster_scores = pd.DataFrame(cell_scores, index=data.obs_names,
                                  columns=list(cluster_genes.keys()))
    data.obsm[f'{groupby}_enrich_scores'] = cluster_scores

    if verbose:
        print(f"Added data.obsm['{groupby}_enrich_scores']")

def coexpr_enrich_labelled(data: sc.AnnData, groupby: str, min_counts: int=2,
                                                            verbose: bool=True):
    """ Coexpression enrichment for cell clusters labelled by gene coexpression.
    """
    ### Converting cluster names to marker list
    cluster_names = np.unique( data.obs[groupby].values )
    cluster_markers = {}
    for cluster in cluster_names:
        cluster_markers[cluster] = cluster.split('-')

    ### Add to anndata so can calculate coexpression scores.
    cluster_marker_key = f'{groupby}_markers'
    data.uns[cluster_marker_key] = cluster_markers
    if verbose:
        print(f"Added {groupby}_markers.")

    ### Now running coexpression scoring.
    coexpr_enrich(data, groupby, cluster_marker_key=cluster_marker_key,
                                         min_counts=min_counts, verbose=verbose)






