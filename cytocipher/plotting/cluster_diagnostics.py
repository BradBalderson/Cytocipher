"""
Diagnostic plots for evaluating clustering.
"""

import sys

import numpy as np
import pandas as pd

import scanpy as sc
from scanpy import AnnData

from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sb
from . import sankey

from scipy.stats import spearmanr

from ..score_and_merge.cluster_merge import average
from ..score_and_merge._group_methods import group_scores
from .cd_helpers import get_pairs, get_p_data, diagnostic_scatter

################################################################################
                    # Coexpression scoring plots #
################################################################################
def enrich_heatmap(data: AnnData, groupby: str, per_cell: bool=True,
                   plot_group: str=None, figsize=(8, 8),
                   dendrogram: bool=False, vmax=1,
                   n_clust_cells: int=50,
                   scale_rows: bool=True, scale_cols: bool=True,
                   show=True
                   ):
    """Plots the enrichment scores for each cluster to show specificity of
        gene coexpression.

        Parameters
        ----------
        data: AnnData
            Single cell data on which enrichment for cluster marker genes has
            been performed.
        groupby: str
            Column in data.obs specifying the cell cluster labels.
        per_cell: bool
            True to plot enrichment scores per cell. False will take average of
            cells within each cluster.
        plot_group: str
            Categorical column in data.obs to group the cells by, this can be
            useful to see if there is some other correspondence between a
            difference set of labels and the clusters scores.
        figsize: tuple
            Size of the figure to plot.
        dendrogram: bool
            Whether to group the cells using the enrichment score based on a
            dendrogram. If False, then it aligns the cell cluster membership
            and the respective clusters such that scores along the diagonal
            indicate scores for cells in their respective cluster.
        vmax: int
            Upper limit on the scores to plot, color scale saturates past this
            point.
        n_clust_cells: int
            The no. of cells to plot per cluster. If less than this, plots
            all cells for a given cluster. If greater than this, than randomly
            samples n_clust_cells from the cluster to plot. Set to None to plot
            all cells, though this can be difficult to visualise if there are
            large group imbalances (which is often the case).
        scale_rows: bool
            Whether to min-max scale the enrichment scores per cell. If scale_cols
            also specified, then scale_cols performed before scale_rows.
        scale_cols: bool
            Whether to min-max scale the enrichment scores per cluster.
            If scale_rows also specified, scale_cols performed before scale_rows.
        show: bool
            Whether to show the plot.
    """
    # Make sure color data transfered! #
    if f'{groupby}_colors' in data.uns:
        colors = data.uns[f'{groupby}_colors']
    else:
        colors = None

    if per_cell and type(n_clust_cells)!=type(None):
        #### Visualising enrichment scores per cell
        #### Do per cell, but subset to x cells per cell type so can see...
        cell_indices = []
        labels = data.obs[groupby].values.astype(str)
        label_set = np.unique(labels)
        for i, label in enumerate(label_set):
            indices = np.where(labels == label)[0]
            cell_indices.extend(
                np.random.choice(indices, min([len(indices), n_clust_cells]),
                                                                 replace=False))

        data = data[cell_indices]

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
    if type(colors)!=type(None):
        score_data.uns[f'{groupby}_colors'] = colors

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
    else:
        return ax

################################################################################
     # Diagnostics pre-testing for significantly different clusters #
################################################################################
def k_optimisation(data: sc.AnnData, groupby: str,
                   show_fit: bool=True, show=True,
                  ):
    """ Plots the results from performing k-optimisation; optimum k shown as
        vertical magenta line!
    """
    #### Retrieving the stored results from optimisation...
    results = data.uns[f'k-opt_{groupby}_results']
    k_opt = results['k_opt']
    ks = results['ks']
    mean_dists = results['mean_dists']
    std_dists = results['std_dists']

    fig, axes = plt.subplots(ncols=1, figsize=(6, 4))
    fig.suptitle("Enrichment score summarisation k-optimisation")

    # axes[0].scatter(ks, mean_dists, color='k')
    # axes[0].set_xlabel("k-value for enrichment summarisation")
    # axes[0].set_ylabel("|orig mean-grouped mean|+.1 * k")
    # axes[0].scatter(ks, std_dists, color='k')
    # axes[0].set_xlabel("k-value for enrichment summarisation")
    # axes[0].set_ylabel("|orig std-grouped std|+.1 * k")
    axes.scatter(ks, std_dists, color='k')
    axes.plot(ks, std_dists, color='plum')
    axes.set_xlabel("k-value for enrichment summarisation")
    axes.set_ylabel("|orig std-grouped std|+.1 * k")

    # if show_fit:
    #     mean_dists_pred = results['mean_dists_pred']
    #     std_dists_pred = results['std_dists_pred']
    #     axes[0].plot(ks, mean_dists_pred, color='deepskyblue')
    #     axes[1].plot(ks, std_dists_pred, color='plum')

    #### Adding labels for optimum K!
    # axes[0].vlines(k_opt, 0, axes[0].get_ylim()[1] / 3, color='magenta')
    # axes[0].text(k_opt, axes[0].get_ylim()[1] / 3, f'k-opt: {k_opt}')
    #
    # axes[1].vlines(k_opt, 0, axes[1].get_ylim()[1] / 3, color='magenta')
    # axes[1].text(k_opt, axes[1].get_ylim()[1] / 3, f'k-opt: {k_opt}')
    axes.vlines(k_opt, axes.get_ylim()[0], axes.get_ylim()[1] /4,
                color='magenta')
    axes.text(k_opt, axes.get_ylim()[1] / 4, f'k-opt: {k_opt}')

    if show:
        plt.show()

def plot_violin(scores: list, score_labels: list, ax=None, show=True):
    """ Violin plot of inputted scores.
    """
    scores_by_cluster = np.array(scores, dtype='object').transpose()

    if type(ax) == type(None):
        fig, ax = plt.subplots(figsize=(5, 5))

    sb.stripplot(data=scores, ax=ax,
                 edgecolor='k', linewidth=1)
    sb.violinplot(data=scores, inner=None, color='.8', ax=ax)
    ax.set_xticklabels(score_labels, rotation=0)

    if show:
        plt.show()

def compare_stats_for_k(data: sc.AnnData, groupby: str, k: int=15,
                        score_group_method: str='quantiles',
                        random_state=20, show=True):
    """ Violin plots of the within cluster enrichment score statistics before
        & after summarisation; checking to ensure these have equivalent
        distributions.
    """
    ###### Getting reuired info
    if score_group_method == 'kmeans' and type(k)!=type(None):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
    else:
        kmeans = None

    ###### Getting the scores...
    labels = data.obs[groupby].values
    enrich_scores = data.obsm[f'{groupby}_enrich_scores']
    label_set = enrich_scores.columns.values

    label_means = np.zeros((len(label_set)))
    label_stds = np.zeros((len(label_set)))
    label_grouped_means = np.zeros((len(label_set)))
    label_grouped_stds = np.zeros((len(label_set)))
    for i, labeli in enumerate(label_set):
        ### The original scores
        label_scores = enrich_scores.values[labels == labeli, i]
        label_means[i] = np.mean(label_scores)
        label_stds[i] = np.std(label_scores)

        ### The grouped scores
        label_scores_grouped = group_scores(label_scores, score_group_method,
                                                                      k, kmeans)
        label_grouped_means[i] = np.mean(label_scores_grouped)
        label_grouped_stds[i] = np.std(label_scores_grouped)

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
    plot_violin([list(label_means), list(label_grouped_means)],
                ['Original means', 'Grouped means'], ax=axes[0], show=False)
    plot_violin([list(label_stds), list(label_grouped_stds)],
                ['Original stds', 'Grouped stds'], ax=axes[1], show=show)

################################################################################
     # Diagnostics after testing for significantly different clusters #
################################################################################
def merge_sankey(data: sc.AnnData, groupby: str, groupby2: str=None,
                aspect: int=5, fontsize: int=8, n_top: int=None,
                 show_range: tuple=None,
                 show_groups: list=None, show_groups2: list=None):
    """ Plots a Sankey diagram indicating which clusters are merged together.

    Parameters
        ----------
        data: AnnData
            Single cell data.
        groupby: str
            Column in data.obs specifying pre-merged clusters input
                                                        to cc.tl.merge_clusters.
        groupby2: str
            Column in data.obs specifying the merged clusters. If not provieded
            assumed to be f'{groupby}_merged'.
        n_top: int
            Specifies the number of merged clusters to show the original clusters
            from which they are derived. The top merged clusters are those with
            the highest number of original clusters which were merged to create
            the merged cluster.
        show_range: tuple
            Which indices in the merge clusters ordered by the number of
            subclusters to show. In format (start, end). Only used if show_groups
            and show_groups2 not specified.
        show_groups: list
            List of labels cells can take in data.obs[groupby] to show how they
            were merged to create groupby2. Only used if show_range and show_groups2
            not specified.
        show_groups2: list
            List of labels cells can take in data.obs[groupby2] to show which
            labels were merged to create these labels from groupby. Only used
            if show_range and show_groups not specified.
    """

    #### Getting colors
    clust1 = groupby
    clust2 = f'{groupby}_merged' if type(groupby2)==type(None) else groupby2

    color_dict = {name: color for name, color in
                  zip(list(data.obs[clust1].cat.categories),
                      data.uns[f'{clust1}_colors'])}
    for name, color in zip(list(data.obs[clust2].cat.categories),
                           data.uns[f'{clust2}_colors']):
        color_dict[name] = color

    #### Getting order of the labels, so can show the clusters being merged
    #### the most first, and the ones the least last.
    clust1_set = np.unique(data.obs[clust1].values.astype(str))
    clust2_set = np.unique(data.obs[clust2].values.astype(str))
    if type(show_groups)!=type(None) and type(show_range)==type(None) and \
        type(show_groups2)==type(None):
        clust1_set = np.array([group for group in show_groups
                               if group in clust1_set])
    if type(show_groups)==type(None) and type(show_range)==type(None) and \
        type(show_groups2)!=type(None):
        clust2_set = np.array([group for group in show_groups2
                               if group in clust2_set])

    clust2_counts = np.zeros((len(clust2_set)))

    subclusts_dict = {}
    for i, clust in enumerate(clust2_set):
        subclusts = np.unique(
            data.obs[clust1].values[data.obs[clust2].values == clust])
        subclusts_dict[clust] = subclusts
        subclust_indices = [np.where(clust1_set == subclust)[0][0]
                            for subclust in subclusts]

        clust2_counts[i] = len(subclusts)

    # clust1_ordered = list(clust1_set[np.argsort(clust1_counts)])
    clust2_ordered = clust2_set[np.argsort(-clust2_counts)]
    if type(n_top)!=type(None):
        clust2_ordered = clust2_ordered[0:n_top]
    elif type(show_range)!=type(None):
        start, end = show_range
        clust2_ordered = clust2_ordered[ list(range(start, end)) ]
    clust2_ordered = list(clust2_ordered)

    clust1_ordered = []
    for clust in clust2_ordered:
        clust1_ordered.extend(subclusts_dict[clust])

    ##### Actually plotting the Sankey diagram.
    original_font = plt.rcParams['font.family'] # Sankey library switches...
    sankey.sankey(data.obs[clust1].values, data.obs[clust2].values,
       aspect=aspect, colorDict=color_dict, fontsize=fontsize,
                 leftLabels=clust1_ordered, rightLabels=clust2_ordered)
    plt.rc('font', family=original_font)  # Switch back to default font

def sig_cluster_diagnostics(data: sc.AnnData, groupby: str,
                            plot_pair: tuple=None,
                            plot_examples: bool=True,
                            skip: int=10, show=True, verbose=True):
    """ Plots violins of enrichment scores compared between clusters with
        with statistics. Useful to diagnose if significance threshold is
        too stringent/relaxed. Note that will throw error if input plot_pair
        were cluster pairs that were not compared if used the MNN heuristic
        for cluster comparison.

        Parameters
        ----------
        data: AnnData
            Single cell data on which cc.tl.merge_clusters has been performed.
        groupby: str
            Column in data.obs specifying pre-merged clusters input
                                                        to cc.tl.merge_clusters.
        plot_pair: tuple
            Input pair of clusters to plot, in format ('1', '2').
        plot_examples: bool
            If true, plots 4 example pairs; representing the upper- and lower-
            bounds of significant versus non-significant cluster pairs.
            Only if plot_pair==None.
        skip: int
            Only if plot_pair==None and plot_examples==None, skip this many
            pairs when plotting cluster pairs from most to least significant in
            difference.
        show: bool
            Whether to show the plot; only works if plot_pair specified.
    """
    mutual_pairs = data.uns[f'{groupby}_mutualpairs']
    ps_dict = data.uns[f'{groupby}_ps']

    labels = data.obs[groupby].values
    enrich_scores = data.obsm[f'{groupby}_enrich_scores']
    label_set = enrich_scores.columns.values
    label_scores = [enrich_scores.values[:, i] for i in range(len(label_set))]
    colors = {name: data.uns[f'{groupby}_colors'][i] for i, name in \
                                    enumerate(data.obs[groupby].cat.categories)}

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

        pairs, pair1s, pair2s = get_pairs(data, groupby)
        for i in range(len(pairs)):
            pair_ = (pair1s[i], pair2s[i]) #tuple(pair_str.split('_'))
            pair_r =  (pair2s[i], pair1s[i]) #tuple(pair_str.split('_')[::-1])

            pair_str = pairs[i]
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
         "Printing top and bottom most significant/non-significant clusters.\n")

    else:
        # Determining order from least to most significant #
        scores = []
        all_pairs = []
        prefixes = []

        pairs, pair1s, pair2s = get_pairs(data, groupby)
        for i in range(len(pairs)):
            pair_ = (pair1s[i], pair2s[i])  # tuple(pair_str.split('_'))
            pair_r = (pair2s[i], pair1s[i])  # tuple(pair_str.split('_')[::-1])

            pair_str = pairs[i]
            pair_str_r = '_'.join(pair_[::-1])

            # Accounting for pair we already saw.
            if pair_ in all_pairs or pair_r in all_pairs:
                continue

            all_pairs.append(pair_)
            if pair_ in mutual_pairs or pair_r in mutual_pairs:
                prefixes.append('Not significant ')
            else:
                prefixes.append('Significant ')

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
                f" to least distinct.\n")

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

def volcano(data: sc.AnnData, groupby: str, p_cut: float=1e-2,
            show_legend: bool=True, legend_loc: str='best',
            figsize: tuple=(6,5), point_size: int=3,
            p_adjust: bool=True, ax: matplotlib.axes.Axes=None,
            highlight_pairs: list=None, highlight_color: str='lime',
            show: bool=True):
    """Plots a Volcano plot showing relationship between logFC of enrichment
        values between clusters and the -log10(p-value) significance.

        Parameters
        ----------
        data: AnnData
            Single cell data on which cc.tl.merge_clusters has been performed.
        groupby: str
            Column in data.obs specifying pre-merged clusters input
                                                        to cc.tl.merge_clusters.
        p_cut: float
            P-value used as cutoff to determine significantly different clusters
        show_legend: bool
            Whether to show the legend that highlights significant versus non-
            significant cluster pairs.
        fig_size: tuple
            Size of figure to plot.
        highlight_pairs: list
            List of pairs to highlight in plot, in order to help set cutoffs.
            In format ['pair1_pari2', 'pair2_pair1'].
        show: bool
            Whether to show the plot.
    """

    labels = data.obs[groupby].values.astype(str)
    enrich_scores = data.obsm[f'{groupby}_enrich_scores']
    label_set = enrich_scores.columns.values.astype(str)

    #### Getting the pairs which were compared
    pvals, log10_ps, pairs, pair1s, pair2s = get_p_data(data, groupby, p_adjust)

    #### Determining the fold-changes
    mean_scores = pd.DataFrame(average(enrich_scores, labels, label_set),
                               index=label_set, columns=label_set)

    fcs = np.array([mean_scores.loc[pair1s[i], pair1s[i]] - mean_scores.loc[
                              pair2s[i], pair1s[i]] for i in range(len(pairs))])

    #### Making the plot
    ylabel = "-log10(p-value)" if not p_adjust else "-log10(adjusted p-value)"
    ax = diagnostic_scatter(fcs, log10_ps, pvals, p_cut, point_size, ax,
                       "log-FC of enrichment scores", ylabel,
                       "Cluster pair comparison statistics",
                       show_legend, figsize, legend_loc, False)

    if type(highlight_pairs)!=type(None):
        pairs_ = [pair for pair in highlight_pairs if pair in pairs]
        if len(pairs_) < len(highlight_pairs):
            print("Warning missing pairs: ", [pair for pair in highlight_pairs
                                              if pair not in pairs_])
        elif len(pairs_)==0:
            print("All highlight pairs not tested.")

        pair_indices = [np.where(pairs==pair)[0][0] for pair in pairs_]
        diagnostic_scatter(fcs[pair_indices], log10_ps[pair_indices],
                           pvals[pair_indices], None, point_size, ax,
                           "log-FC of enrichment scores", ylabel,
                           "Cluster pair comparison statistics",
                           False, figsize, legend_loc, False,
                           point_color=highlight_color)

    if show:
        plt.show()
    else:
        return ax

def check_abundance_bias(data: sc.AnnData, groupby: str, p_cut: float=1e-2,
                         show_legend: bool=True, legend_loc: str='best',
                         figsize: tuple=(6,4), point_size: int=3,
                         p_adjust: bool=True, ax: matplotlib.axes.Axes=None,
                         show: bool=True):
    """ Checks for bias between pair significance and the DIFFERENCE in the
        number of cells in each cluster being compared.
        Spearman correlation displayed. Should be no bias, i.e. correlation
        close to 0.

        Parameters
        ----------
        data: AnnData
            Single cell data on which cc.tl.merge_clusters has been performed.
        groupby: str
            Column in data.obs specifying pre-merged clusters input
                                                        to cc.tl.merge_clusters.
        p_cut: float
           P-value used as cutoff to determine significantly different clusters.
           If None then just shows the relationship without highlighting the
           significant versus non-significant cluster pairs.
        show_legend: bool
            Whether to show the legend that highlights significant versus non-
            significant cluster pairs.
        show: bool
            Whether to show the plot.
    """

    labels = data.obs[groupby].values.astype(str)
    label_set = np.unique( labels )

    #### Getting the pairs which were compared
    pvals, log10_ps, pairs, pair1s, pair2s = get_p_data(data, groupby, p_adjust)

    ### Calculating log-FCs for pairs tested.
    cell_counts = np.array([len(np.where(labels == label)[0])
                                                        for label in label_set])

    count_fcs = np.array([np.log2(cell_counts[label_set == pair1s[i]])[0] -
                          np.log2(cell_counts[label_set == pair2s[i]])[0]
                          for i in range(len(pairs))])

    #### Making the plot
    ylabel = "-log10(p-value)" if not p_adjust else "-log10(adjusted p-value)"
    diagnostic_scatter(count_fcs, log10_ps, pvals, p_cut, point_size, ax,
                       "log-FC of cell counts between clusters", ylabel,
                       "DIFFERENCE in cluster pair cell abundance bias",
                       show_legend, figsize, legend_loc, show, show_corr=True)

def check_total_abundance_bias(data: sc.AnnData, groupby: str, p_cut: float=1e-2,
                         show_legend: bool=True, legend_loc: str='best',
                         figsize: tuple=(6,4), point_size: int=3,
                         p_adjust: bool=True, ax: matplotlib.axes.Axes=None,
                               show: bool=True):
    """ Checks for bias between pair significance and the TOTAL number of cells
        in the pair of clusters being compared.
        Different from cc.pl.check_abundance_bias in that this does not check
        for bias due to imbalanced groups, but rather the total number of cells
        considered as observations.
        Spearman correlation displayed.
        Should be no bias, i.e. correlation close to 0.

        Parameters
        ----------
        data: AnnData
            Single cell data on which cc.tl.merge_clusters has been performed.
        groupby: str
            Column in data.obs specifying pre-merged clusters input
                                                        to cc.tl.merge_clusters.
        p_cut: float
           P-value used as cutoff to determine significantly different clusters.
           If None then just shows the relationship without highlighting the
           significant versus non-significant cluster pairs.
        show_legend: bool
            Whether to show the legend that highlights significant versus non-
            significant cluster pairs.
        ax: Axes
            Matplotlib axes on which to plot.
        show: bool
            Whether to show the plot.
    """

    labels = data.obs[groupby].values.astype(str)
    label_set = np.unique(labels)

    #### Getting the pairs which were compared
    pvals, log10_ps, pairs, pair1s, pair2s = get_p_data(data, groupby, p_adjust)

    ### Calculating log-FCs for pairs tested.
    cell_counts = np.array([len(np.where(labels == label)[0])
                                                        for label in label_set])
    log_counts = np.array(
        [np.log2(np.sum([cell_counts[label_set == pair1s[i]][0],
                         cell_counts[label_set == pair2s[i]][0]]))
         for i in range(len(pairs))])

    #### Making the plot
    ylabel = "-log10(p-value)" if not p_adjust else "-log10(adjusted p-value)"
    diagnostic_scatter(log_counts, log10_ps, pvals, p_cut, point_size, ax,
                       "log2-cell counts in cluster pair", ylabel,
                       "TOTAL cluster pair cell abundance bias",
                       show_legend, figsize, legend_loc, show, show_corr=True)

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
