"""
Helper functions for the cluster diagnostics; mostly to prevent code duplication
and make long-term maintence easier.
"""

import sys
import scanpy as sc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

def get_pairs(data: sc.AnnData, groupby: str):
    """ Gets the cluster pair information in a safe way; i.e. not sensitive to
        '_' in the cluster names.
    """
    pairs = np.array( list(data.uns[f'{groupby}_ps'].keys()) )

    # Old method sensitive to cluster names with '_' in name.
    # pair1s = np.array([pair.split('_')[-1] for pair in pairs])
    # pair2s = np.array([pair.split('_')[0] for pair in pairs])

    # Pair names processed so not sensitive to '_' in input names
    clusters = np.unique( data.obs[groupby].values )
    max_clust_len = max( [len(clust) for clust in clusters] )

    pair1s = [' ' * max_clust_len] * len(pairs)
    pair2s = [' ' * max_clust_len] * len(pairs)
    for clust1 in clusters:
        for clust2 in clusters:
            indices = np.where(pairs == f'{clust2}_{clust1}')[0]
            if len(indices) > 0:
                index = indices[0]
                pair1s[index] = clust1
                pair2s[index] = clust2

    return pairs, pair1s, pair2s

def get_p_data(data: sc.AnnData, groupby: str, p_adjust: bool=False):
    """ Retrives log10 p-values and the respective cluster pairs.

    Parameters
    ----------
    p_adjust: bool
        True if want to plot the adjusted p-values, False for unadjusted, and
        None to plot automatically determine which to plot.
    """

    suffix = 'ps' if not p_adjust else 'padjs'

    pvals = np.array(list(data.uns[f'{groupby}_{suffix}'].values()))
    min_sig_nonzero = min(pvals[pvals > 0])
    log10_ps = np.array([-np.log10(pval + sys.float_info.min)
                                                             for pval in pvals])
    log10_ps[pvals == 0] = -np.log10(min_sig_nonzero)

    pairs, pair1s, pair2s = get_pairs(data, groupby)

    return pvals, log10_ps, pairs, pair1s, pair2s

def diagnostic_scatter(xs: np.array, log10_ps: np.array,
                       pvals: np.array, p_cut: float, point_size: float,
                        ax: matplotlib.axes.Axes,
                       xlabel: str, ylabel: str, title: str,
                       show_legend: bool, figsize: tuple, legend_loc: str,
                       show: bool, show_corr: bool=False,
                       point_color: str='orchid'):
    """Scatter plot comparing significance levels to some other metric, e.g.
        log-FC, difference in cell numbers between pairs, log cell abundance.
    """

    if type(ax)==type(None):
        fig, ax = plt.subplots(figsize=figsize)

    ##### Making the scatter plot
    if type(p_cut)!=type(None):
        sig_bool = pvals < p_cut
        nonsig_bool = pvals >= p_cut

        ax.scatter(xs[sig_bool], log10_ps[sig_bool],
                   s=point_size, c='dodgerblue', )
        ax.scatter(xs[nonsig_bool], log10_ps[nonsig_bool],
                   s=point_size, c='red', )
        ax.hlines(-np.log10(p_cut), ax.get_xlim()[0], ax.get_xlim()[1],
                  colors='red')

        legend = [f'Significant pairs ({sum(sig_bool)})',
                  f'Non-significant pairs ({sum(nonsig_bool)})']
    else:
        ax.scatter(xs, log10_ps, s=point_size, c=point_color)

        legend = [f'Cluster pairs ({len(xs)})']

    ax.set_xlabel( xlabel )
    ax.set_ylabel( ylabel )
    ax.set_title( title )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(legend, loc=legend_loc)

    if show_corr:
        corr = round(spearmanr(xs, log10_ps)[0], 3)
        ax.text((ax.get_xlim()[0] + np.min(xs)) / 2, np.max(log10_ps),
                f'ρ: {corr}', c='k')

    if show:
        plt.show()

    return ax






