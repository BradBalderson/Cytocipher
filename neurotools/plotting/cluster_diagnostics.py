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
    """Plots the enrichment scores for each cluster to show specificity of
        gene coexpression.
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

################################################################################
     # Diagnostics after testing for significantly different clusters #
################################################################################
def sig_cluster_diagnostics(data: sc.AnnData, groupby: str,
                            plot_pair: tuple=None,
                            plot_examples: bool=True,
                            skip: int=10, show=True, verbose=True):
    """ Plots violins of enrichment scores compared between clusters with
        with statistics. Useful to diagnose if significance threshold is
        too stringent/relaxed.

        Parameters
        ----------
        data: AnnData     Single cell data on which nts.tl.merge_clusters has
                                                                 been performed.
        groupby: str    Column in data.obs specifying pre-merged clusters input
                                                       to nts.tl.merge_clusters.
        plot_pair: tuple     Input pair of clusters to plot, in format ('1', '2').
        plot_examples: bool If true, plots 4 example pairs; representing the
                            upper- and lower- bounds of significant versus
                            non-significant cluster pairs. Only if plot_pair==None.
        skip: int   Only if plot_pair==None and plot_examples==None,
                    skip this many pairs when plotting cluster pairs from most
                    to least significant in difference.
        show: bool  Whether to show the plot; only works if plot_pair specified.
    """
    mutual_pairs = data.uns[f'{groupby}_mutualpairs']
    ps_dict = data.uns[f'{groupby}_ps']

    labels = data.obs[groupby].values
    enrich_scores = data.obsm[f'{groupby}_enrich_scores']
    label_set = enrich_scores.columns.values
    label_scores = [enrich_scores.values[:, i] for i in range(len(label_set))]

    ################## Determining which pairs to plot #########################
    if type(plot_pair) != type(None):
        pairs_to_plot = [plot_pair]
        print_names = [f'Input pair {plot_pair}']
        skip = 1

    elif plot_examples:
        print_names = ['Top non-significant', 'Bottom non-significnat',
                       'Top significant', 'Bottom significant']
        pairs_to_plot = ['', '', '',
                         '']  # top_nonsig, bottom_nonsig, top_sig, bottom_sig
        scores = [0, 1, 1, 0]
        for pair_str in ps_dict:
            pair_ = tuple(pair_str.split('_'))
            pair_r = tuple(pair_str.split('_')[::-1])

            pair_str_r = '_'.join(pair_[::-1])
            p_ = ps_dict[pair_str]
            p_r = ps_dict[pair_str_r]

            score = np.mean([p_, p_r])

            if (pair_ in mutual_pairs or pair_r in mutual_pairs) and \
                    score > scores[0]:
                pairs_to_plot[0] = pair_
                scores[0] = score
            if (pair_ in mutual_pairs or pair_r in mutual_pairs) and \
                    score < scores[1]:
                pairs_to_plot[1] = pair_
                scores[1] = score
            if (pair_ not in mutual_pairs and pair_r not in mutual_pairs) and \
                    score < scores[2]:
                pairs_to_plot[2] = pair_
                scores[2] = score
            if (pair_ not in mutual_pairs and pair_r not in mutual_pairs) and \
                    score >= scores[3]:
                pairs_to_plot[3] = pair_
                scores[3] = score

        # Actually, want to plot in different order
        order = [2, 3, 0, 1]
        print_names = np.array(print_names)[order]
        pairs_to_plot = np.array(pairs_to_plot)[order]
        skip = 1

        if verbose:
            print(
           "Printing top and bottom most significant/non-significant clusters.")

    else:
        # Determining order from least to most significant #
        scores = []
        all_pairs = []
        prefixes = []
        for pair_str in ps_dict:
            pair_ = tuple(pair_str.split('_'))
            pair_r = tuple(pair_str.split('_')[::-1])

            # Accounting for pair we already saw.
            if pair_ in all_pairs or pair_r in all_pairs:
                continue

            all_pairs.append(pair_)
            if pair_ in mutual_pairs or pair_r in mutual_pairs:
                prefixes.append('Not significant ')
            else:
                prefixes.append('Significant ')

            pair_str_r = '_'.join(pair_[::-1])
            p_ = ps_dict[pair_str]
            if pair_str_r in ps_dict:  # Accounting if comparison not made.
                p_r = ps_dict[pair_str_r]
            else:
                p_r = 1

            scores.append(np.mean([p_, p_r]))
        scores = np.array(scores)
        order = np.argsort(scores)

        prefixes = np.array(prefixes)[order]
        pairs_to_plot = np.array(all_pairs)[order]
        print_names = [f'{prefixes[_k_]}: Pair rank {_k_}' for _k_ in
                       range(len(pairs_to_plot))]

        skip = skip if skip < len(pairs_to_plot) else 1
        if verbose:
            print(
                f"Plotting every {skip} pairs in order of most distinct clusters"
                f" to least distinct.")

    ################## Determining which pairs to plot #########################
    for i_ in range(0, len(pairs_to_plot), skip):
        pairi = pairs_to_plot[i_]

        i, labeli = np.where(label_set == pairi[0])[0][0], pairi[0]
        j, labelj = np.where(label_set == pairi[1])[0][0], pairi[1]

        pair_label = '_'.join(pairi)
        pair_label_r = '_'.join(pairi[::-1])
        print(print_names[i_])
        print(
            f'p={ps_dict[pair_label_r]} ({pairi[0]} cells; {pairi[0]} scores) vs '
            f'({pairi[1]} cells; {pairi[0]} scores)')
        print(
            f'p={ps_dict[pair_label]} ({pairi[0]} cells; {pairi[1]} scores) vs '
            f'({pairi[1]} cells; {pairi[1]} scores)')

        labeli_labelj_scores = label_scores[j][labels == labeli]
        labelj_labelj_scores = label_scores[j][labels == labelj]

        labeli_labeli_scores = label_scores[i][labels == labeli]
        labelj_labeli_scores = label_scores[i][labels == labelj]

        scores_by_cluster = np.array([
                                     labeli_labeli_scores, labelj_labeli_scores,
                                    labeli_labelj_scores, labelj_labelj_scores],
                                     dtype='object').transpose()

        colors_ = [colors[pairi[0]], colors[pairi[1]],
                   colors[pairi[0]], colors[pairi[1]], ]
        colors_2 = [colors[pairi[0]], colors[pairi[0]],
                    colors[pairi[1]], colors[pairi[1]], ]

        fig, ax = plt.subplots(figsize=(5, 5))
        sb.stripplot(data=scores_by_cluster, ax=ax, palette=colors_,
                     edgecolor='k', linewidth=1)
        sb.violinplot(data=scores_by_cluster, inner=None, color='.8', ax=ax,
                      palette=colors_2)
        ax.set_xticklabels([f'{pairi[0]} cells; {pairi[0]} scores',
                            f'{pairi[1]} cells; {pairi[0]} scores',
                            f'{pairi[0]} cells; {pairi[1]} scores',
                            f'{pairi[1]} cells; {pairi[1]} scores',
                            ],
                           rotation=40
                           )
        ax.set_xlabel(groupby)
        ax.set_ylabel("Cluster enrich scores")
        # Removing boxes outside #
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if show or type(plot_pair) == type(None):
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

################################################################################
                        # No longer in use #
################################################################################
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
