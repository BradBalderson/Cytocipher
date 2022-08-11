"""
General plots which can be used across different methods for diagnostics.
"""

import numpy as np
import pandas as pd
import upsetplot
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

def get_upset_df(obj_lists, group_names):
    """ Creates the necessary input to draw an upset plot for visualising overlaps
        between multiple groups!
    Args:
        obj_lists (list<list<object>>): List of items in different groups want \
                                          to generate upset-plot for to compare.
        group_names (list<str>): List of strings indicating the names for the \
                                                               different groups.
    Returns:
        pd.DataFrame: This is a dataframe formatted in the required formatted \
                    for input to upsetplot so can visualise multi-overlaps.
    """
    all_hs_genes = []
    [all_hs_genes.extend(de_hs_) for de_hs_ in obj_lists]
    all_hs_genes = np.unique(all_hs_genes)

    de_hs_genes = obj_lists
    samples = group_names

    de_hs_vals = np.zeros((len(samples), len(all_hs_genes)))
    for i, samp in enumerate(samples):
        for j, gene in enumerate(all_hs_genes):
            if gene in de_hs_genes[i]:
                de_hs_vals[i, j] = 1
    de_hs_df = pd.DataFrame(de_hs_vals.transpose(),
                            index=all_hs_genes, columns=samples)

    upset_df = pd.DataFrame()
    col_names = samples
    for idx, col in enumerate(de_hs_df[samples]):
        temp = []
        for i in de_hs_df[col]:
            if i != 0:
                temp.append(True)
            else:
                temp.append(False)
        upset_df[col_names[idx]] = temp

    upset_df['c'] = 1
    example = upset_df.groupby(col_names).count().sort_values('c')

    return example

def upset_plot(obj_lists, group_names=None, fig_title='', min_subset_size=1,
               sort_by="cardinality", sort_groups_by=None, show=True):
    """ Creates an upset plot, a visualisation which is useful for comparing \
        overlaps between multiple groups when have more than one group.

    Args:
        obj_lists (list<list<object>>): List of items in different groups want \
                                          to generate upset-plot for to compare.
        group_names (list<str>): List of strings indicating the names for the \
                                                               different groups.
    """
    obj_lists = obj_lists[::-1]
    if type(group_names)==type(None):
        group_names = [f'group_{i}' for i in range(len(obj_lists))]
    else:
        group_names = group_names[::-1]

    upset_df = get_upset_df(obj_lists, group_names)

    upsetplot.plot(upset_df['c'], sort_by=sort_by,
                   sort_categories_by=sort_groups_by,
                   min_subset_size=min_subset_size)
    plt.title(fig_title, loc='left')
    if show:
        plt.show()

def rank_scatter(
    items,
    y,
    y_label: str = "",
    x_label: str = "",
    highlight_items=None,
    show_text=True,
    color="gold",
    alpha=0.5,
    lr_text_fp=None,
    axis_text_fp=None,
    ax=None,
    show=True,
    highlight_color="red",
    rot: float = 90,
    point_sizes: np.array = None,
    pad=0.2,
    figsize=None,
    width_ratio=7.5 / 50,
    height=4,
    point_size_name="Sizes",
    point_size_exp=2,
    show_all: bool = False,
):
    """General plotting function for showing ranked list of items."""
    ranks = np.array(list(range(len(items))))

    highlight = type(highlight_items) != type(None)
    if type(lr_text_fp) == type(None):
        lr_text_fp = {"weight": "bold", "size": 8}
    if type(axis_text_fp) == type(None):
        axis_text_fp = {"weight": "bold", "size": 12}

    if type(ax) == type(None):
        if type(figsize) == type(None):
            width = width_ratio * len(ranks) if show_text and not highlight else 7.5
            if width > 20:
                width = 20
            figsize = (width, height)
        fig, ax = plt.subplots(figsize=figsize)

    # Plotting the points #
    scatter = ax.scatter(
        ranks,
        y,
        alpha=alpha,
        c=color,
        s=None if type(point_sizes) == type(None) else point_sizes ** point_size_exp,
        edgecolors="none",
    )
    y_min, y_max = ax.get_ylim()
    y_max = y_max + y_max * pad
    ax.set_ylim(y_min, y_max)
    if type(point_sizes) != type(None):
        # produce a legend with a cross section of sizes from the scatter
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=4)
        [handle.set_markeredgecolor("none") for handle in handles]
        starts = [label.find("{") for label in labels]
        ends = [label.find("}") + 1 for label in labels]
        sizes = [
            float(label[(starts[i] + 1) : (ends[i] - 1)])
            for i, label in enumerate(labels)
        ]
        counts = [int(size ** (1 / point_size_exp)) for size in sizes]
        labels2 = [
            label.replace(label[(starts[i]) : (ends[i])], "{" + str(counts[i]) + "}")
            for i, label in enumerate(labels)
        ]
        legend2 = ax.legend(
            handles,
            labels2,
            frameon=False,
            # bbox_to_anchor=(0.1, 0.05, 1., 1.),
            handletextpad=1.6,
            loc="upper right",
            title=point_size_name,
        )

    if show_text:
        if highlight:
            ranks_ = ranks[[np.where(items == item)[0][0] for item in highlight_items]]
            ax.scatter(
                ranks_,
                y[ranks_],
                alpha=alpha,
                c=highlight_color,
                s=None
                if type(point_sizes) == type(None)
                else (point_sizes[ranks_] ** point_size_exp),
                edgecolors=color,
            )
            ranks = ranks_ if not show_all else ranks

        for i in ranks:
            ax.text(i - 0.2, y[i], items[i], rotation=rot, fontdict=lr_text_fp)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(x_label, axis_text_fp)
    ax.set_ylabel(y_label, axis_text_fp)

    if show:
        plt.show()
    else:
        return ax

def plot_go(go_results, n_top=12, lr_text_fp=None, highlight_go=None,
            highlight_color=None, figsize=(6,4.8), rot=45, show=True,
            max_text=25, point_size_exp: float=2):
    """Plots go terms outputted from beautifulcells.tools.go.go.
    """

    gos = go_results.loc[:, "Description"].values.astype(str)
    y = -np.log10(go_results.loc[:, "p.adjust"].values)
    sizes = go_results.loc[:, "Count"].values
    rank_scatter(
        gos[0:n_top],
        y[0:n_top],
        point_sizes=sizes[0:n_top],
        highlight_items=highlight_go,
        lr_text_fp=lr_text_fp,
        highlight_color=highlight_color,
        figsize=figsize,
        y_label="-log10(padjs)",
        x_label="GO Rank",
        height=6,
        color="deepskyblue",
        rot=rot,
        width_ratio=0.4,
        show=show,
        point_size_name="n-genes",
        point_size_exp=point_size_exp,
        show_all=n_top <= max_text,
    )




