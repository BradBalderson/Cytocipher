"""
Helper functions for clustering.
"""

import numpy as np
import pandas as pd
import scanpy as sc

def n_different(data, groupby, pair, logfc_cutoff, padj_cutoff):
    """ Gets no. of significant DE between inputted cell types.
    """
    ##### Calling de genes
    sc.tl.rank_genes_groups(data, groupby=groupby, use_raw=False,
                            groups=pair[0:1], reference=pair[1],
                            )

    #### Getting results
    logfcs_rank = pd.DataFrame(
        data.uns['rank_genes_groups']['logfoldchanges']
    ).values[:, 0].astype(float)
    padjs_rank = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj']
                              ).values[:, 0].astype(float)

    up_bool = np.logical_and(logfcs_rank > logfc_cutoff,
                             padjs_rank < padj_cutoff)
    down_bool = np.logical_and(logfcs_rank < -logfc_cutoff,
                               padjs_rank < padj_cutoff)
    # n_de = len(np.where(sig_bool)[0])
    n_up = len(np.where(up_bool)[0])
    n_down = len(np.where(down_bool)[0])
    n_de = sum([n_up, n_down])

    return n_de



