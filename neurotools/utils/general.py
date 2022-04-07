import numpy as np
import pandas as pd

from numba import njit, prange

def summarise_data(data, labels,
                       label_set: np.array=None, average: bool=True):
    """ Formats the data such that the rows are ordered according to the \
            appearence of values in colors, and if average is true, then will \
            get an average of each labelled column.

    Args:
        data (pandas.DataFrame): Cells * Genes.

        labels (numpy.array<str>): Label for each cell.

        label_set (list-like<str> or None): Specifies order of the labels.

        average (bool or str): Whether to return average of each label. If \
                                        string can be 'avg', 'median', or 'sum'.

    Returns:
        pandas.DataFrame, numpy.array<str>: \
                    The data formatted as described above, and the labels for \
                    each cell, but ordered same as in colors.
    """
    summary_function = None
    if average == True or average == 'avg' or average == 'average':
        summary_function = np.mean

    elif average == 'median':
        summary_function = np.median

    elif average == 'sum':
        summary_function = sum

    label_set = label_set if type(label_set)!=type(None) else np.unique(labels)

    # Need to summarise value for each gene #
    if type(summary_function) != type(None):
        formatted_data = np.zeros((len(label_set), data.shape[1]))
        for i, label in enumerate(label_set):
            label_indices = np.where(labels == label)[0]
            vals = data.values[label_indices, :]
            label_avg_exprs = np.apply_along_axis(summary_function, 0, vals)
            formatted_data[i, :] = label_avg_exprs

        labels_ordered = label_set

    else:  # Just need to re-order to ensure cells appropriately grouped #
        label_indices = []
        for label in label_set:
            label_indices.extend( list(np.where(labels == label)[0]) )
        formatted_data = data.values[:, label_indices]
        labels_ordered = labels[label_indices]

    formatted_data = pd.DataFrame(formatted_data, index=labels_ordered,
                                                         columns=data.columns)

    return formatted_data

@njit(parallel=True)
def summarise_data_fast(data, label_indices):
    formatted_data = np.zeros((len(label_indices), data.shape[1]),
                                                                dtype=np.float_)
    for i in prange(len(label_indices)):
        for j in range(data.shape[1]):
            formatted_data[i, j] = np.median( data[label_indices[i], j] )

    return formatted_data

