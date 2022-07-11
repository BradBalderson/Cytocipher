"""
General functions for comparing sets cell labellings.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

def get_overlap_counts(data: sc.AnnData, groupby1: str, groupby2: str,
                       uns_key: str = 'annot_overlaps',
                       verbose: bool = True):
    """ Calculates no. of cells belonging to the different categories of groupby1 and groupby2.
    """

    labels = data.obs[groupby1].values.astype(str)
    label_set = np.array(list(data.obs[groupby1].cat.categories))

    labels2 = data.obs[groupby2].values.astype(str)
    label_set2 = np.array(list(data.obs[groupby2].cat.categories))

    overlap_counts = np.zeros((len(label_set), len(label_set2)))
    for i, label in enumerate(label_set):
        label_bool = labels == label
        for j, label2 in enumerate(label_set2):
            label_bool2 = labels2 == label2

            overlap_counts[i, j] = len(
                np.where(np.logical_and(label_bool, label_bool2))[0])

    overlap_counts_df = pd.DataFrame(overlap_counts, index=label_set,
                                     columns=label_set2)
    data.uns[uns_key] = overlap_counts_df
    if verbose:
        print(f"Added data.uns['{uns_key}']")


def annot_overlap_barplot(data: sc.AnnData, groupby1: str, groupby2: str,
                          uns_key: str = 'annot_overlaps',
                          bar_width: float = .85,
                          figsize=(10, 5), scale: bool = True,
                          transpose: bool = False,
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
                color=colors[label], edgecolor='white',
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





