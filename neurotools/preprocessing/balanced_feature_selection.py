"""
Implements balanced feature selection approach; whereby use marker genes for
different cell types to select additional genes in order to perform clustering
on user's variation of interest.
"""

import os
import numpy as np
import pandas as pd
from scanpy import AnnData

from numba import njit

def load_nps():
    """ Loads the neuropeptidergic/dopaminergic genes.
    """
    path = os.path.dirname(os.path.realpath(__file__))
    nps = pd.read_csv(path+'/../dbs/NP-DA_markers.txt', header=None).values[:, 0]
    return nps

@njit
def pearson_r(vals1, vals2):
    """ Calculates Pearson's correlation coefficient.
    """
    mean1, mean2 = np.mean(vals1), np.mean(vals2)
    diff1 = vals1 - mean1
    diff2 = vals2 - mean2
    std1, std2 = np.std(vals1), np.std(vals2)

    return np.mean( (diff1/std1)*(diff2/std2) )

def balanced_feature_select(data: AnnData, reference_genes: np.array,
                            n_total: int=500, verbose=bool):
    """Performs balanced feature selection using inputted genes as reference
       to select additional genes which displays similar expression patterns.
    """
    expr_vals = data.X.toarray()
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

    ########## Adding results to AnnData #############
    ## Getting the information in shape to reference all genes ##
    selected_indices = [np.where(expr_genes==gene)[0][0] for gene in selected]
    selected_bool = np.full((data.shape[1]), False, dtype=np.bool_)
    selected_bool[selected_indices] = True

    selected_corrs_full = np.zeros((data.shape[1]), dtype=np.float_)
    selected_corrs_full[selected_indices] = selected_corrs

    selected_match_full = np.empty((data.shape[1]), dtype=expr_genes.dtype)
    selected_match_full[selected_indices] = selected_match

    data.var['bfs_selected'] = selected_bool
    data.var['bfs_corrs'] = selected_corrs_full
    data.var['bfs_matched'] = selected_match_full

    if verbose:
        print("Added data.var['bfs_selected'].")
        print("Added data.var['bfs_corrs'].")
        print("Added data.var['bfs_matched']")



