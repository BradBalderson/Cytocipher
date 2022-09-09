"""
Methods for aggregate the cell scores before calling differential cluster pairs.
This deals with the problem of inflated p-values & bias toward cluster pairs
with more groups.
"""

import numpy as np

def group_scores(labeli_labelj_scores: np.array, method: str,
                                                    k: int, kmeans_obj: object):
    """ Groups the scores according to the desired method.

    Parameters
    ----------
    method: str
        One of 'kmeans', 'quantile_bin', & 'quantiles'.
    k: int
        Number of score groups to create. If None, then just returns original
        scores. If len(labeli_labelj_scores) <= k; then also return original.
    """
    ### Account for n cells <= k and when user doesn't want to aggregate.
    if type(k)==type(None) or len(labeli_labelj_scores) <= k:
        return labeli_labelj_scores

    elif method=='kmeans':
        return kmeans_group(labeli_labelj_scores, kmeans_obj)

    elif method=='quantile_bin':
        return quantile_bin(labeli_labelj_scores, k)

    elif method=='quantiles':
        return quantiles(labeli_labelj_scores, k)

    else:
        raise Exception(f"For score_group_method, got {method} but expected one"
                        f" of ['kmeans', 'quantile_bin', 'quantiles'].")

def kmeans_group(labeli_labelj_scores: np.array, kmeans_obj: object):
    """ Groups by first performing KMeans on the scores, then averages cells
        falling into the same KMeans group.
    """
    groupsi = kmeans_obj.fit_predict( labeli_labelj_scores.reshape(-1, 1) )
    labeli_labelj_scores_mean = [np.mean(labeli_labelj_scores[groupsi == k])
                                 for k in np.unique(groupsi)]

    return labeli_labelj_scores_mean

def quantile_bin(labeli_labelj_scores: np.array, k: int):
    """ Bins the cells by the indicated number of quantiles and takes the
        average of the cell scores falling into each bin.
    """
    order = np.argsort(labeli_labelj_scores)
    scores_ordered = labeli_labelj_scores[order]

    interval = len(labeli_labelj_scores) // k

    start = 0
    labeli_labelj_scores_binned = []
    for end in range(interval, len(scores_ordered), interval):
        vals__ = scores_ordered[start:end]
        labeli_labelj_scores_binned.append(np.mean(vals__))
        start = end

    if end < len(scores_ordered):
        labeli_labelj_scores_binned.append(np.mean(scores_ordered[end:]))

    return labeli_labelj_scores_binned

def quantiles(labeli_labelj_scores: np.array, k: int):
    """ Simply takes the quantiles of the cells themselves, without any
        averaging, in order to summarise the data...
    """
    q = np.array([(1 / k) * i for i in range(1, k + 1)])
    labeli_labelj_quantiles = np.quantile(labeli_labelj_scores, q=q,
                                                               method='nearest')
    return labeli_labelj_quantiles

