"""
Implements balanced feature selection approach; whereby use marker genes for
different cell types to select additional genes in order to perform clustering
on user's variation of interest.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

from .bbknn_functions import bbknn_modified
from .bfs_helpers import pearson_r,  get_selected_dist, get_selected_corr, \
                         add_selected_to_anndata, calc_coexpr_odds, odds_cutoff

def load_nps():
    """ Loads the neuropeptidergic/dopaminergic genes.
    """
    path = os.path.dirname(os.path.realpath(__file__))
    nps = pd.read_csv(path+'/../dbs/NP-DA_markers.txt', header=None).values[:, 0]
    return nps

def load_sex():
    """Loads sex DE genes from development papers"""
    path = os.path.dirname(os.path.realpath(__file__))
    sex_df = pd.read_csv(path + '/../dbs/sex_genes.txt', sep='\t', index_col=0)
    return sex_df

def balanced_feature_select_graph(data: AnnData, reference_genes: np.array,
                                  n_total: int=500,
                                  pca: bool=False, n_pcs: int=100,
                                  recompute_pca: bool=True,
                                  use_annoy: bool=False,
                                  metric: str='correlation',
                                  approx: bool=True,
                                  neighbors_within_batch: int=None,
                                  bg_size: int=10000,
                                  padj_method: str='fdr_bh',
                                  padj_cutoff: float=.01,
                                  verbose: bool=True,
                                  ):
    """ Performs balanced feature selection, except instead of using Pearson
        correlation across all genes (which won't scale well), performs more
        similar to BBKNN except each "batch" is actually a gene, & the
        neighbours we are getting are the neighbouring genes, & when we get the
        closest neighbourhood genes we make sure they aren't already selected.
        NOTE: Settings of use_annoy=False, metric='correlation' produces results
                almost identical to balanced_feature_select_v2.
    """
    if type(neighbors_within_batch)==type(None):
        neighbors_within_batch = n_total

    ###### Deciding which features we will compute the neighbourhood graph on
    if pca and ('X_pca' not in data.varm or recompute_pca):
        genes_data = sc.AnnData(data.to_df().transpose())
        sc.tl.pca(genes_data, n_comps=n_pcs)

        gene_features = genes_data.obsm['X_pca']
        data.varm['X_pca'] = gene_features
        if verbose:
            print("Added data.varm['X_pca']")

    elif 'X_pca' in data.varm:
        gene_features = data.varm['X_pca']

    else:
        expr = data.X if type(data.X)==np.ndarray else data.X.toarray()
        gene_features = expr.transpose()

    # Let's make each of the reference genes a batch #
    batch_list = np.array(["-1"] * data.shape[1])
    ref_indices = [np.where(data.var_names.values == ref_gene)[0][0]
                   for ref_gene in reference_genes]
    batch_list[ref_indices] = "1"

    ####### Getting the nearest neighbours & their resepective distances
    if verbose:
        print("Getting distances between reference genes & remaining genes...")
    knn_indices, knn_distances = bbknn_modified(gene_features,
                                             batch_list, use_annoy=use_annoy,
                                             n_pcs=gene_features.shape[1],
                                             metric=metric,
                                             approx=approx,
                                             neighbors_within_batch=
                                                         neighbors_within_batch)

    # Let's check if the ranked genes have similar correlations #
    graph_refs = data.var_names.values[batch_list == "1"]  # was out of order!!!

    # Getting genes to select per reference gene #
    expr_genes = data.var_names.values.astype(str)

    # Now getting the balanced selected genes #
    selected, selected_corrs, selected_match = get_selected_dist(
                                                    knn_indices,
                                                    knn_distances, graph_refs,
                                                    expr_genes, n_total,
                                                    verbose)

    # Adding the results to AnnData #
    add_selected_to_anndata(data, expr_genes, selected, selected_corrs,
                            selected_match, verbose)

    # Post-hoc metric of method performance; odds-score
    if verbose:
        print("Calculating odds-score of coexpression between selected genes & reference genes..")
    calc_coexpr_odds(data, verbose=verbose)

    # Using the odds-score & randomisation to determine cutoff
    if verbose:
        print(
            "Dynamically determining odds-score based cutoff using random genes..")
    odds_cutoff(data, bg_size=bg_size, verbose=verbose,
                padj_method=padj_method, padj_cutoff=padj_cutoff)

def balanced_feature_select_v2(data: AnnData, reference_genes: np.array,
                            n_total: int=500, verbose: bool=True):
    """Similar to the development version, except in this case we don't select
    one gene for each reference gene per iteration, instead select all genes
    for reference at once. Is fater than the original, but still not fast
    enough.
    """
    expr_vals = data.X if type(data.X)==np.ndarray else data.X.toarray()
    expr_genes = np.array(data.var_names, dtype=str)

    # Getting locations of the reference & non-reference genes #
    ref_indices = [np.where(expr_genes == ref_gene)[0][0]
                   for ref_gene in reference_genes]
    not_ref_indices = [i for i in list(range(len(expr_genes)))
                       if i not in ref_indices]
    not_ref_genes = expr_genes[not_ref_indices]

    expr_not_ref = expr_vals[:, not_ref_indices]

    # Correlations between ref genes (rows) & non-ref genes (cols)
    corrs = np.zeros((len(ref_indices), len(not_ref_indices)))
    for i, ref_index in enumerate(ref_indices):
        ref_expr = expr_vals[:, ref_index]
        ref_corrs = np.apply_along_axis(pearson_r, 0, expr_not_ref, ref_expr)

        corrs[i, :] = ref_corrs

    ##### Take the best correlated with each ref until n_total selected ########
    selected, selected_corrs, selected_match = get_selected_corr(corrs,
                                                    reference_genes, expr_genes,
                                                         not_ref_genes, n_total)

    # Adding the results to AnnData #
    add_selected_to_anndata(data, expr_genes, selected, selected_corrs,
                            selected_match, verbose)

    # Post-hoc metric of method performance; odds-score
    if verbose:
        print(
            "Calculating odds-score of coexpression between selected genes & reference genes..")
    calc_coexpr_odds(data, verbose=verbose)

    # Using the odds-score & randomisation to determine cutoff
    if verbose:
        print(
            "Dynamically determining odds-score based cutoff using random genes..")
    odds_cutoff(data, verbose=verbose)

def balanced_feature_select_original(data: AnnData, reference_genes: np.array,
                                     n_total: int=500, verbose=bool):
    """ Original implementation from Development paper.
        Too slow for big data.
    """
    expr_vals = data.X if type(data.X) == np.ndarray else data.X.toarray()
    expr_genes = np.array(data.var_names, dtype=str)

    # Getting locations of the reference & non-reference genes #
    ref_indices = [np.where(expr_genes == ref_gene)[0][0]
                   for ref_gene in reference_genes]
    not_ref_indices = [i for i in list(range(len(expr_genes)))
                       if i not in ref_indices]
    not_ref_genes = expr_genes[not_ref_indices]

    expr_not_ref = expr_vals[:, not_ref_indices]

    # Correlations between ref genes (rows) & non-ref genes (cols)
    corrs = np.zeros((len(ref_indices), len(not_ref_indices)))
    for i, ref_index in enumerate(ref_indices):
        ref_expr = expr_vals[:, ref_index]
        ref_corrs = np.apply_along_axis(pearson_r, 0, expr_not_ref, ref_expr)

        corrs[i, :] = ref_corrs

    ##### Take the best correlated with each ref until n_total selected ########
    # Determining max_gene length for making array #
    selected = np.empty((n_total), dtype=expr_genes.dtype)
    for i, ref in enumerate(reference_genes):
        selected[i] = ref
    selected_corrs = np.ones((n_total))
    selected_match = selected.copy()  # Genes the selected match with
    remaining_indices = np.zeros((corrs.shape[1]), dtype=np.int64)
    for i in range(corrs.shape[1]):
        remaining_indices[i] = i
    ref_index = 0
    n_selected = len(reference_genes)
    while n_selected < n_total:
        corrs_sub = corrs[:, remaining_indices]
        values = corrs_sub[ref_index, :]
        order = np.argsort(-values)
        sorted = values[order]
        best_corr, best_i = sorted[0], order[0]
        selected_corrs[n_selected] = best_corr
        selected[n_selected] = not_ref_genes[remaining_indices][best_i]
        selected_match[n_selected] = reference_genes[ref_index]

        index_bool = np.full((len(remaining_indices)), True, dtype=np.bool_)
        index_bool[best_i] = False
        remaining_indices = remaining_indices[index_bool]

        if ref_index == len(reference_genes) - 1:
            ref_index = 0
        else:
            ref_index += 1

        n_selected += 1

        # Adding the results to AnnData #
        add_selected_to_anndata(data, expr_genes, selected, selected_corrs,
                                selected_match, verbose)

        # Post-hoc metric of method performance; odds-score
        if verbose:
            print(
                "Calculating odds-score of coexpression between selected genes & reference genes..")
        calc_coexpr_odds(data, verbose)

""" Junk Code
def bfs_run(data: AnnData, reference_genes: np.array,
            method: str="graphNN", n_total: int=500, verbose: bool=True):
    Key function running the different balanced feature selection methods.
    
    method_to_function = {"graphNN": balanced_feature_select_graph,
                          "v2": balanced_feature_select_v2,
                          "original": balanced_feature_select_original}
    if method not in method_to_function:
        raise Exception(f"Method must be: {list(method_to_function.keys())}")

    bfs_function = method_to_function[method]
    bfs_function(data, reference_genes, n_total=n_total, verbose=verbose)
"""

