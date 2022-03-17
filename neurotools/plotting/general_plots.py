"""
General plots which can be used across different methods for diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt

fp = {'weight': 'bold', 'size': 12}

def distrib(x, bins=100, x_label='', fig_title='', log=False, density=False,
            figsize=(6.4,4.8), add_mean=False, logbase=np.e,
            color='blue', alpha=1, ax=None, fig=None, show=True, cutoff=None,
            cutoff_color='r',
            label='', total=None, return_total=False, ylims=None, xlims=None):
    """Plots a histogram of values."""
    if type(ax)==type(None) or type(fig)==type(None):
        fig, ax = plt.subplots(figsize=figsize)

    # Getting the counts in desired format #
    counts, bins = np.histogram(x, bins=bins)
    logcounts = np.log(counts+1)/np.log(logbase) if log else counts
    if density and type(total)==type(None):
        total = sum(logcounts)
        logcounts = logcounts/total
    elif density:
        logcounts = logcounts/total

    ax.hist(bins[:-1], bins, weights=logcounts, color=color, alpha=alpha,
            label=label)
    ax.set_xlabel(x_label, fp)
    if not density:
        ax.set_ylabel(f'log{round(logbase, 2)}-counts' if log else 'counts', fp)
    else:
        ax.set_ylabel('density-'+f'log{round(logbase, 2)}-counts'
                                                      if log else 'density', fp)
    fig.suptitle(fig_title)

    if add_mean:
        mean = np.mean(x)
        y = ax.get_ylim()[1]*.5
        ax.vlines(mean, 0, y, colors=cutoff_color)
        ax.text(mean, y, f'mean:{round(mean, 4)}', c=cutoff_color)
    if cutoff:
        y = ax.get_ylim()[1] * .5
        ax.vlines(cutoff, 0, y, colors=cutoff_color)
        ax.text(cutoff, y, f'cutoff: {round(cutoff, 4)}', c=cutoff_color)

    # Add axes these limits #
    if type(xlims)!=type(None):
        ax.set_xlim(*xlims)
    if type(ylims)!=type(None):
        ax.set_ylim(*ylims)

    # Removing boxes outside #
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show:
        plt.show()
    elif not return_total:
        return fig, ax
    else:
        return fig, ax, total






