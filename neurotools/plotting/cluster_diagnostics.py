"""
Diagnostic plots for evaluating clustering.
"""

import numpy as np
import pandas as pd

import scanpy as sc
from scanpy import AnnData

from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt
import seaborn as sb

def enrich_heatmap(data: AnnData, groupby: str, per_cell: bool=True,
                   plot_group: str=None, figsize=(12, 12),
                   dendrogram: bool=False, vmax=1, show=True,
                   scale_rows: bool=False, scale_cols: bool=False):
    """Plots the Giotto enrichment scores for each clusters own Limma_DE genes to show
        specificity of gene coexpression in cluster.
    """
    cell_scores_df = data.obsm[f'{groupby}_enrich_scores']

    ##### Handling scale, only min-max implemented.
    expr_scores = cell_scores_df.values
    if scale_cols:
        expr_scores = minmax_scale(expr_scores, axis=0)  # per enrich scale
    if scale_rows:
        expr_scores = minmax_scale(expr_scores, axis=1) # per cell scale
    if scale_rows or scale_cols:
        cell_scores_df = pd.DataFrame(expr_scores, index=cell_scores_df.index,
                                                 columns=cell_scores_df.columns)

    score_data = sc.AnnData(cell_scores_df, obs=data.obs)
    if dendrogram:
        sc.tl.dendrogram(score_data, groupby, use_rep='X')

    if type(plot_group)!=type(None):
        groupby = plot_group

    if per_cell:
        ax = sc.pl.heatmap(score_data, score_data.var_names, figsize=figsize,
                           groupby=groupby, show_gene_labels=True, vmax=vmax,
                           show=False, dendrogram=dendrogram)
        ax['heatmap_ax'].set_title("Cluster Limma_DE gene coexpression score",
                                   fontdict={'fontweight': 'bold',
                                             'fontsize': 20})

    else:
        ax = sc.pl.matrixplot(score_data, score_data.var_names, figsize=figsize,
                         groupby=groupby, dendrogram=dendrogram, vmax=vmax,
                            #show_gene_labels=True,
                              show=False
                         )

    if show:
        plt.show()

def plot_cluster_sils(data: AnnData, groupby: str):
    """ Plots the silhouette scores per cluster!
    """
    scores = data.uns[f'{groupby}_sils'][0]
    cluster_labels = data.uns[f'{groupby}_sils'][1]

    scores_by_cluster = np.array(
                    [list(scores[cluster_labels == cluster])
          for cluster in np.unique(cluster_labels)], dtype='object').transpose()
    fig, ax = plt.subplots(figsize=(10, 5))
    sb.stripplot(data=scores_by_cluster, ax=ax)
    sb.violinplot(data=scores_by_cluster, inner=None, color='.8', ax=ax)
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_xlabel(groupby)
    ax.set_ylabel("Cell Silohouette score")
    # Removing boxes outside #
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=np.mean(scores))
    plt.show()





