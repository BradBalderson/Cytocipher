"""
Diagnostic plots for evaluating clustering.
"""

import numpy as np
import pandas as pd

import scanpy as sc
from scanpy import AnnData

import matplotlib.pyplot as plt

def enrich_heatmap(data: AnnData, groupby: str):
    """Plots the Giotto enrichment scores for each clusters own DE genes to show
        specificity of gene coexpression in cluster.
    """
    cell_scores_df = data.obsm[f'{groupby}_enrich_scores']
    score_data = sc.AnnData(cell_scores_df, obs=data.obs)

    ax = sc.pl.heatmap(score_data, score_data.var_names, figsize=(12, 12),
                       groupby=groupby, show_gene_labels=True,
                       show=False)
    ax['heatmap_ax'].set_title("Cluster DE gene coexpression score",
                               fontdict={'fontweight': 'bold',
                                         'fontsize': 20})
    plt.show()







