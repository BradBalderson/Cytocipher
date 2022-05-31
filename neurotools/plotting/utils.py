"""
Utility visualisation functions.
"""

import numpy as np

import scanpy as sc

import matplotlib
import matplotlib as plt

def get_colors(labels, cmap, rgb=False):
    """ Gets colors.
    """
    # Determining the set of labels #
    label_set = np.unique(labels)

    # Initialising the ordered dict #
    cellTypeColors = {}

    # Ordering the cells according to their frequency and obtaining colors #
    nLabels = len(label_set)
    cmap = plt.cm.get_cmap(cmap, nLabels)
    rgbs = [cmap(i)[:3] for i in range(nLabels)]
    # rgbs = list(numpy.array(rgbs)[order]) # Make sure color order is the same.

    # Populating the color dictionary with rgb values or hexcodes #
    for i in range(len(label_set)):
        cellType = label_set[i]
        rgbi = rgbs[i]
        if not rgb:
            cellTypeColors[cellType] = matplotlib.colors.rgb2hex(rgbi)
        else:
            cellTypeColors[cellType] = rgbi

    return cellTypeColors


def add_colors(data, groupby, cmap='tab20'):
    """ Adds colors for the groupby, unlike scanpy which will cap colours
        past a certain large number of clusters.
    """
    color_dict = get_colors(data.obs[groupby].values.astype(str), cmap)
    data.uns[f'{groupby}_colors'] = [color_dict[cluster] for cluster in
                                               data.obs[groupby].cat.categories]




