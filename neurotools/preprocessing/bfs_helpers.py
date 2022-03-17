"""
Helper functions for the BFS method.
"""

import numpy as np
import pandas as pd
from scanpy import AnnData
from numba import njit

from tqdm import tqdm

from statsmodels.stats.multitest import multipletests

@njit
def pearson_r(vals1, vals2):
    """ Calculates Pearson's correlation coefficient.
    """
    mean1, mean2 = np.mean(vals1), np.mean(vals2)
    diff1 = vals1 - mean1
    diff2 = vals2 - mean2
    std1, std2 = np.std(vals1), np.std(vals2)

    return np.mean( (diff1/std1)*(diff2/std2) )

def genes_per_feature(corrs: np.ndarray, n_total: int, dist: bool=False):
    """Determines the no. of genes to give to each feature.

    Parameters
    ----------
    corrs: np.ndarray
        Reference genes * genes distance or correlations
    n_total: int
        The number of genes to select in total.
    dist: bool
        Whether or not values in corrs represent distances or correlations.
    Returns
    -------
    n_genes_per_ref: np.array
        Array of integers specifying the no. of genes to select for each feature.
    """
    n_ref_genes = corrs.shape[0]
    # Determining how many genes per feature #
    n_left = n_total -  n_ref_genes # Remainder after keeping input genes
    n_genes = n_left //  n_ref_genes
    remaining = n_left - ( n_ref_genes*n_genes)
    # Give left overs to weakest genes
    mean_corrs = np.nanmean(corrs, axis=1)
    mean_corrs = mean_corrs if not dist else -mean_corrs # Ensure correct order.
    order = np.argsort(mean_corrs)
    n_genes_per_ref = np.full((n_ref_genes), n_genes, dtype=np.int64)
    n_genes_per_ref[order[0:remaining]] += 1

    return n_genes_per_ref

def get_selected_dist(knn_indices: np.ndarray, knn_distances: np.ndarray,
                      reference_genes: np.array, expr_genes: np.array,
                      n_total: int, verbose: bool):
    """Performs the core component of getting the balanced feature per reference
        feature.
    """
    selected = np.empty((n_total), dtype=expr_genes.dtype)
    for i, ref in enumerate(reference_genes):
        selected[i] = ref
    selected_corrs = np.ones((n_total))
    selected_match = selected.copy()  # Genes the selected match with
    # Determining how many genes per feature #
    n_genes_per_ref = genes_per_feature(knn_distances, n_total, dist=True)

    with tqdm(
            total=len(reference_genes),
            desc="Performing balanced selection of genes based on reference genes...",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
            disable=verbose == False,
    ) as pbar:
        n_selected = len(reference_genes)
        for ref_index in range(len(reference_genes)):
            ref_n_genes = n_genes_per_ref[ref_index]
            order = knn_indices[ref_index,:]
            order_sub, i = [], 0
            while len(order_sub) < ref_n_genes:
                if expr_genes[order[i]] not in selected:
                    order_sub.append(i)
                i += 1
            order_sub = np.array(order_sub)
            best_is = order_sub[0:ref_n_genes]
            best_corrs = knn_distances[ref_index, best_is]

            end_index = n_selected + ref_n_genes
            selected_corrs[n_selected:end_index] = best_corrs
            selected[n_selected:end_index] = expr_genes[order][best_is]
            selected_match[n_selected:end_index] = reference_genes[ref_index]

            pbar.update(1)

            n_selected = end_index

    return selected, selected_corrs, selected_match

def get_selected_corr(corrs: np.ndarray, reference_genes: np.array,
                 expr_genes: np.array, not_ref_genes: np.array,
                 n_total: int):
    """ Gets the balanced selection features based on correlation.
    """
    selected = np.empty((n_total), dtype=expr_genes.dtype)
    for i, ref in enumerate(reference_genes):
        selected[i] = ref
    selected_corrs = np.ones((n_total))
    selected_match = selected.copy()  # Genes the selected match with
    # Determining how many genes per feature #
    n_genes_per_ref = genes_per_feature(corrs, n_total)

    n_selected = len(reference_genes)
    for ref_index in range(len(reference_genes)):
        ref_n_genes = n_genes_per_ref[ref_index]
        values = corrs[ref_index, :]
        order = np.argsort(-values)
        order_sub, i = [], 0
        while len(order_sub)<ref_n_genes:
            if not_ref_genes[order[i]] not in selected:
                order_sub.append( order[i] )
            i += 1
        order_sub = np.array( order_sub )
        best_is = order_sub[0:ref_n_genes]
        best_corrs = values[best_is]

        end_index = n_selected + ref_n_genes
        selected_corrs[n_selected:end_index] = best_corrs
        selected[n_selected:end_index] = not_ref_genes[best_is]
        selected_match[n_selected:end_index] = reference_genes[ref_index]

        n_selected = end_index

    return selected, selected_corrs, selected_match

def add_selected_to_anndata(data: AnnData, expr_genes: np.array,
                            selected: np.array, selected_corrs: np.array,
                            selected_match: np.array, verbose: bool=True):
    """Adds selected gene information to AnnData
    """
    reference_genes = np.unique(selected_match)
    reference_indices = [np.where(expr_genes == gene)[0][0]
                         for gene in reference_genes]
    reference_bool = np.full((data.shape[1]), False, dtype=np.bool_)
    reference_bool[reference_indices] = True

    selected_indices = [np.where(expr_genes == gene)[0][0] for gene in selected]
    selected_bool = np.full((data.shape[1]), False, dtype=np.bool_)
    selected_bool[selected_indices] = True

    selected_corrs_full = np.zeros((data.shape[1]), dtype=np.float_)
    selected_corrs_full[selected_indices] = selected_corrs

    selected_match_full = np.empty((data.shape[1]), dtype=expr_genes.dtype)
    selected_match_full[selected_indices] = selected_match

    bfs_results = pd.DataFrame(index=data.var_names.values,
                           columns=['bfs_reference',
                               'bfs_selected', 'bfs_corrs', 'bfs_matched'])
    bfs_results['bfs_reference'] = reference_bool
    bfs_results['bfs_selected'] = selected_bool
    bfs_results['bfs_corrs'] = selected_corrs_full
    bfs_results['bfs_matched'] = selected_match_full

    data.varm['bfs_results'] = bfs_results

    if verbose:
        print("Added data.varm['bfs_results'].")

def calc_coexpr_odds(data: AnnData,
                     selected_woRef: np.array=None,
                     selected_match_woRef: np.array=None,
                     return_odds: bool=False,
                     verbose: bool=True):
    """Calculates the logodds of coexpression between inputted genes &
        reference genes.
    """
    # Getting desired information from AnnData #
    reference_bool = data.varm['bfs_results']['bfs_reference'].values

    # If we don't have inputted data, retrieve it from the AnnData #
    if type(selected_woRef)==type(None) and \
                                         type(selected_match_woRef)==type(None):
        reference_genes = data.var_names.values[reference_bool]
        selected_bool = data.varm['bfs_results']['bfs_selected'].values
        selected = data.var_names.values[selected_bool]
        selected_match = data.varm['bfs_results']['bfs_matched'].values[
                                                                  selected_bool]
        selected_woRef = [sel for sel in selected if sel not in reference_genes]
        selected_match_woRef = [match for i, match in enumerate(selected_match)
                                if selected[i] not in reference_genes]

    reference_genes = np.unique(selected_match_woRef)
    expr_genes = data.var_names.astype(str)

    expr = data.X if type(data.X) == np.ndarray else data.X.toarray()
    ref_expr_bool = expr[:, [np.where(expr_genes == ref)[0][0]
                                                for ref in reference_genes]] > 0
    selected_expr_bool = expr[:, [np.where(expr_genes == sel)[0][0]
                                  for sel in
                                  selected_woRef]] > 0
    selected_expr_rates = selected_expr_bool.sum(axis=0) / expr.shape[0]

    ref_indices = [np.where(reference_genes == match)[0][0] for match in
                                                           selected_match_woRef]
    ref_expr_rates = ref_expr_bool.sum(axis=0) / expr.shape[0]
    ref_expr_rates_duplicate = ref_expr_rates[ref_indices]

    ref_expr_bool_duplicates = ref_expr_bool[:, ref_indices]
    coexpr_bool = np.logical_and(ref_expr_bool_duplicates, selected_expr_bool)
    coexpr_rates = coexpr_bool.sum(axis=0) / expr.shape[0]

    rand_coexpr_rates = ref_expr_rates_duplicate * selected_expr_rates
    odds = coexpr_rates / rand_coexpr_rates
    #logodds = odds #np.log2(odds + (1 / expr.shape[0]))

    # Getting indices of where to place odds scores #
    if not return_odds:
        odds_indices = [np.where(data.var_names.values==gene)[0][0]
                        for gene in selected_woRef]
        odds_full = np.zeros((data.shape[1]))
        odds_full[odds_indices] = odds
        odds_full[reference_bool] = max(odds) # Set the reference genes as max
                                # Could calculate odds for reference but will
                                # inflate the odds score distrib. for downstream
                                # plotting.


        data.varm['bfs_results']['odds'] = odds_full
        if verbose:
            print("Added data.varm['bfs_results']['odds']")

        return

    return odds # Mostly used for random odds

def odds_cutoff(data: AnnData, bg_size: int=10000,
                padj_method: str='fdr_bh', padj_cutoff: float=.01,
                verbose: bool=True):
    """Determines cutoff based on random background using the odds scores &
        random matching of the reference genes with background genes.
    """
    #### Retrieving required information #####
    all_genes = data.var_names.values
    reference_bool = data.varm['bfs_results']['bfs_reference'].values
    reference_genes = all_genes[reference_bool]
    selected_bool = data.varm['bfs_results']['bfs_selected'].values
    selected = all_genes[selected_bool]

    bg_genes = [gene for gene in data.var_names if gene not in selected]
    rand_genes = np.random.choice(bg_genes, size=bg_size, replace=False)
    rand_matches = np.random.choice(reference_genes, size=bg_size)
    rand_odds = calc_coexpr_odds(data, selected_woRef=rand_genes,
                                      selected_match_woRef=rand_matches,
                                      return_odds=True)

    """ Now let's dynamically determine the cutoff based on p-values....
    """
    odds_indices = \
             np.where(np.logical_and(selected_bool, reference_bool == False))[0]
    odds = data.varm['bfs_results']['odds'].values[odds_indices]
    ps = np.array(
        [len(np.where(rand_odds > odd)[0]) / len(rand_odds) for odd in odds])
    ps[ps == 0] = 1 / len(rand_odds)

    padjs = multipletests(ps, method=padj_method)[1]

    ############ Summarising information to save to data ###########################
    # Creating dataframe of the random values #
    bg = pd.DataFrame([rand_genes, rand_matches, rand_odds],
                      index=['bg_genes', 'rand_reference', 'rand_odds']
                      ).transpose()
    # Ensure can save h5ad
    bg['rand_odds'] = bg['rand_odds'].astype(float)
    bg.index = bg.index.values.astype(str)

    ps_full = np.ones((data.shape[1]))
    padjs_full = ps_full.copy()
    sig_odds_full = np.zeros((data.shape[1]))

    ps_full[odds_indices] = ps
    ps_full[reference_bool] = 0
    padjs_full[odds_indices] = padjs
    padjs_full[reference_bool] = 0

    sig_full = padjs_full < padj_cutoff
    sig_odds_full[odds_indices] = odds
    sig_odds_full[reference_bool] = max(odds)
    sig_odds_full[sig_full == False] = 0

    ############## Saving information to AnnData ##################################
    data.uns['bfs_background'] = bg

    data.varm['bfs_results']['ps'] = ps_full
    data.varm['bfs_results']['padjs'] = padjs_full
    data.varm['bfs_results']['sig'] = sig_full
    data.varm['bfs_results']['sig_odds'] = sig_odds_full # max of this >0 is cutoff

    data.var['highly_variable'] = sig_full

    if verbose:
        print("Added data.uns['bfs_background']")
        print("Added data.varm['bfs_results']['ps']")
        print("Added data.varm['bfs_results']['padjs']")
        print("Added data.varm['bfs_results']['sig']")
        print("Added data.varm['bfs_results']['sig_odds']")
        print("Added data.var['highly_variable']")
