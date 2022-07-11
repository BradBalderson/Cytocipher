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




