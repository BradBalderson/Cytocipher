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

################################################################################
                    # Coexpression scoring plots #
################################################################################

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
        ax['heatmap_ax'].set_title("Cluster enrich scores",
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

################################################################################
                    # Comparing cluster plots #
################################################################################
def annot_overlap_barplot(data: sc.AnnData, groupby1: str, groupby2: str,
                          uns_key: str = 'annot_overlaps',
                          bar_width: float = .85,
                          figsize=(10, 5), scale: bool = True,
                          transpose: bool = False, edge_color=None,
                          remove_labelling: bool = False):
    """Plots relationship between annotations by showing proportions of each type found within
        each of the other categories.
    """
    if uns_key not in data.uns:
        print("Need to run get_overlap_counts() first.")

    overlap_counts_df = data.uns[uns_key]
    if transpose:  ##### Looking at relationship in opposite direction.
        overlap_counts_df = overlap_counts_df.transpose()
        tmp = groupby1
        groupby1 = groupby2
        groupby2 = tmp

    labelset1 = overlap_counts_df.index.values
    labelset2 = overlap_counts_df.columns.values

    ##### Getting colors
    if f'{groupby2}_colors' in data.uns:
        colors = {val: data.uns[f'{groupby2}_colors'][i]
                  for i, val in enumerate(data.obs[groupby2].cat.categories)}
    else:
        colors = {val: 'grey' for
                  i, val in enumerate(data.obs[groupby2].cat.categories)}

    ##### Scaling the data to proportions.
    if scale:
        total_counts = overlap_counts_df.sum(axis=1)
        overlap_counts = np.apply_along_axis(np.divide, 0,
                                             overlap_counts_df.values,
                                             total_counts)
    else:
        overlap_counts = overlap_counts_df.values

    ##### Creating the barplots...
    fig, axs = plt.subplots(figsize=figsize)
    col_locs = list(range(len(labelset1)))
    for labeli, label in enumerate(labelset2):
        props = overlap_counts[:, labeli]

        if labeli == 0:
            bottom = [0] * len(props)

        axs.bar(col_locs, props, bottom=bottom,
                color=colors[label], edgecolor=edge_color,
                width=bar_width, )
        bottom = list(np.array(bottom) + props)

        # Custom x axis
        axs.set_xticks(list(range(len(labelset1))), list(labelset1),
                       rotation=90)
        axs.spines['right'].set_visible(False)
        axs.spines['top'].set_visible(False)
        if remove_labelling:
            axs.set_xticks([])
            axs.set_yticks([])
            axs.spines['left'].set_visible(False)
            axs.spines['bottom'].set_visible(False)
        else:
            axs.set_xlabel(f"{groupby1} Clusters")

def annot_overlap_heatmap(data: sc.AnnData, overlap_key: str,
                          transpose: bool=False, show: bool=True):
    """Plots a heatmap of the number of cells overlapping two sets of anotations.
    """
    overlap_counts = minmax_scale(data.uns[overlap_key].values, axis=0)
    overlap_counts = pd.DataFrame(overlap_counts,
                                  index=data.uns[overlap_key].index,
                                  columns=data.uns[overlap_key].columns)

    if transpose:
        overlap_counts = overlap_counts.transpose()

    cluster_overlaps = sc.AnnData(overlap_counts)
    cluster_overlaps.obs['sc_clusters'] = cluster_overlaps.obs_names.values
    cluster_overlaps.obs['sc_clusters'] = cluster_overlaps.obs[
                                               'sc_clusters'].astype('category')
    sc.pl.matrixplot(cluster_overlaps,
                     var_names=cluster_overlaps.var_names.values,
                     groupby='sc_clusters', show=show)


