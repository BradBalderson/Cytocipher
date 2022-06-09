"""
Functions for measuring quality of clusters.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

import numba
from numba.typed import List
from numba import njit, prange

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

    ### Accounting for case where might have only one marker gene !!
    if expr.shape[1] < min_counts:
        min_counts = expr.shape[1]

    ### Must be coexpression of atleast min_count markers!
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

@njit(parallel=True)
def get_enrich_scores(full_expr: np.ndarray, all_genes: np.array,
                      cluster_genes_List: np.array,
                      min_counts: int,
                      ):
    """ Gets the enrichment of the cluster-specific gene combinations in each
        individual cell.
    """
    cell_scores = np.zeros((full_expr.shape[0], len(cluster_genes_List)))
    for i in prange( len(cluster_genes_List) ):
        genes_ = cluster_genes_List[i]
        gene_indices = np.zeros( genes_.shape )
        for gene_index, gene in enumerate( genes_ ):
            gene_indices[gene_index] = np.where(all_genes == gene)[0][0]

        cluster_scores_ = coexpr_score(full_expr[:, gene_indices],
                                       min_counts=min_counts)
        cell_scores[:, i] = cluster_scores_

    return cell_scores

def coexpr_enrich(data: sc.AnnData, groupby: str,
                  cluster_marker_key: str = None,
                  min_counts: int = 2, n_cpus: int=1,
                  verbose: bool = True):
    """ NOTE: unlike the giotto function version, this one assumes have already done DE.
    """
    # Setting threads for paralellisation #
    if type(n_cpus) != type(None):
        numba.set_num_threads( n_cpus )

    if type(cluster_marker_key) == type(None):
        cluster_marker_key = f'{groupby}_markers'

    cluster_genes_dict = data.uns[cluster_marker_key]

    # Putting all genes into array for speed.
    all_genes = []
    [all_genes.extend(cluster_genes_dict[cluster])
                                              for cluster in cluster_genes_dict]
    # Getting correct typing
    str_dtype = f"<U{max([len(gene_name) for gene_name in all_genes])}"
    all_genes = np.unique( all_genes ).astype(str_dtype)

    #### Need to convert the markers into a Numba compatible format, easiest is
    #### List of numpy arrays.
    cluster_genes_List = List()
    for cluster in cluster_genes_dict:
        #### Genes stratified by cluster
        cluster_genes = np.array([gene for gene in cluster_genes_dict[cluster]],
                                                                dtype=str_dtype)
        cluster_genes_List.append( cluster_genes )

    full_expr = data[:, all_genes].X.toarray()

    ###### Getting the enrichment scores...
    cell_scores = get_enrich_scores(full_expr, all_genes,
                                    cluster_genes_List, min_counts)

    ###### Adding to AnnData
    cluster_scores = pd.DataFrame(cell_scores, index=data.obs_names,
                                  columns=list(cluster_genes_dict.keys()))
    data.obsm[f'{groupby}_enrich_scores'] = cluster_scores

    if verbose:
        print(f"Added data.obsm['{groupby}_enrich_scores']")

def coexpr_enrich_labelled(data: sc.AnnData, groupby: str, min_counts: int=2,
                                             n_cpus: int=1, verbose: bool=True):
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
                                         min_counts=min_counts, verbose=verbose,
                                                                 n_cpus=n_cpus,)



################################################################################
                        # Currently not in use #
################################################################################
# TODO could be good to use this in giotto_enrich above...
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





