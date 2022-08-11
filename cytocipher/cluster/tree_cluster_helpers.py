"""
Helper functions for the tree-based clustering.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

def is_sig_different(data: AnnData, groupby: str, label1: str, label2: str,
                        max_de: int, padj_cutoff: float, logfc_cutoff: float):
    """ Checks if two inputted labels are significantly
        different from one another.

    NOTE: this should consider logfc as an absolute cutoff, right now only
        considers if Limma_DE in label1 versus label2, but it might DOWN regulate
        relative to the other label!!!
    """

    ##### Calling de genes
    sc.tl.rank_genes_groups(data, groupby=groupby, use_raw=False,
                            groups=[label1], reference=label2,
                            )

    #### Getting results
    logfcs_rank = pd.DataFrame(data.uns['rank_genes_groups']['logfoldchanges']
                                                    ).values[:, 0].astype(float)
    padjs_rank = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj']
                                                    ).values[:, 0].astype(float)

    # TODO find better way to deal with this problem & figure out why happening!
    if np.all(np.isnan(logfcs_rank)):
        #raise Exception("All logfcs nan!!!!")
        scores_rank = pd.DataFrame(data.uns['rank_genes_groups']['scores']
                                                    ).values[:, 0].astype(float)
        logfcs_rank = np.zeros(logfcs_rank.shape)
        logfcs_rank[scores_rank>0] = logfc_cutoff + 1

    sig_bool = np.logical_and(logfcs_rank > logfc_cutoff,
                              padjs_rank < padj_cutoff)
    n_de = len(np.where(sig_bool)[0])

    # If have less than this amount of Limma_DE genes, not sig different
    return n_de > max_de, n_de

def is_sig_different_scores(data: AnnData, groupby: str, label1: str, label2: str,
                        max_de: int, padj_cutoff: float,
                            #score_cutoff: float
                            ):
    """ Checks if two inputted labels are significantly
        different from one another.
    """

    ##### Calling de genes
    sc.tl.rank_genes_groups(data, groupby=groupby, use_raw=False,
                            groups=[label1], reference=label2,
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
    n_de = len(np.where( padjs_rank < padj_cutoff )[0])

    # If have less than this amount of Limma_DE genes, not sig different
    return n_de > max_de, n_de


