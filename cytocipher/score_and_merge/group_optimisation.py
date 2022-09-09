"""
Methods for optimising the grouping parameter k to ensure the original
    enrichment score distribution is not distorted, while minimising the
    number of observations used for the test (& thereby correlation between
    test significance and the number of cells compared!)
"""

import numpy as np
import pandas as pd
import scanpy as sc

import scipy.optimize as optim
from sklearn.cluster import KMeans

from ._group_methods import group_scores

def get_summary_dist(enrich_scores, labels, label_set,
                     score_group_method, k, kmeans=None):
    """ Gets the distance between the summary statistics
                                            before & after metric summarisation.
    """
    label_means = np.zeros((len(label_set)))
    label_stds = np.zeros((len(label_set)))
    label_grouped_means = np.zeros((len(label_set)))
    label_grouped_stds = np.zeros((len(label_set)))
    for i, labeli in enumerate(label_set):

        ### The original scores
        label_scores = enrich_scores.values[labels==labeli, i]
        label_means[i] = np.mean(label_scores)
        label_stds[i] = np.std(label_scores)

        ### The grouped scores
        label_scores_grouped = group_scores(label_scores,
                                            score_group_method, k, kmeans)
        label_grouped_means[i] = np.mean(label_scores_grouped)
        label_grouped_stds[i] = np.std(label_scores_grouped)

    ##### Plotting...
    #mean_dist = np.linalg.norm((label_means-label_grouped_means))
    #std_dist = np.linalg.norm(label_stds-label_grouped_stds)
    mean_dist = (np.abs( np.mean(label_means)-np.mean(label_grouped_means) )+.1)*k
    std_dist = (np.abs( np.mean(label_stds)-np.mean(label_grouped_stds))+.1)*k
    return mean_dist, std_dist

def logistic(k, a, b, c, d):
    """ Function which defines relationship between K and distances.
    """
    return a*(b**(k)) + (c / ( np.exp(d*k)) )

def optimise_k(data: sc.AnnData, groupby:str,
               score_group_method: str='quantiles', random_state: int=20,
               k_start: int=5, k_end: int=400, k_step: int=10,
               verbose: bool=True):
    """ Optimises the parameter K to ensure key statistics of original
                                            enrichment scores are not distorted.
    """
    labels = data.obs[groupby].values
    enrich_scores = data.obsm[f'{groupby}_enrich_scores']
    label_set = enrich_scores.columns.values

    if score_group_method == 'kmeans':
        kmeans = KMeans(n_clusters=k, random_state=random_state)
    else:
        kmeans = None

    # For different values of K, find point where have minimal K but best
    # estimate of the distribution....
    ks = list( range(k_start, k_end, k_step) )
    mean_dists = []
    std_dists = []
    for k in ks:
        mean_dist, std_dist = get_summary_dist(enrich_scores, labels, label_set,
                                               score_group_method, k,
                                               kmeans=kmeans)

        mean_dists.append( mean_dist )
        std_dists.append( std_dist )

    ############## Fitting function to determine optimal K #####################
    # Set min bounds on all coefficients, and set different max bounds for each
    bounds = (-10000, [1000000., 100000, 1000000000., 1000000])
    # Randomly initialize the coefficients
    p0 = np.random.exponential(size=4)

    # (a_mean, b_mean, c_mean, d_mean), cov = optim.curve_fit(logistic, ks,
    #                                                         mean_dists,
    #                                                         bounds=bounds,
    #                                                         p0=p0)
    # (a_std, b_std, c_std, d_std), cov = optim.curve_fit(logistic, ks, std_dists,
    #                                                     bounds=bounds, p0=p0)
    #
    # mean_dists_pred = [logistic(k, a_mean, b_mean, c_mean, d_mean) for k in ks]
    # std_dists_pred = [logistic(k, a_std, b_std, c_std, d_std) for k in ks]

    ################## Storing results for optimum K ###########################
    ### Just realised optimising with respect to mean makes no sense, since
    ### smaller K will give more accurate mean always, because K=1 is mean!
    #mean_diff = np.max(mean_dists) - np.min(mean_dists)
    #std_diff = np.max(std_dists) - np.min(std_dists)
    #mean_dists_ = (np.array(mean_dists) - np.min(mean_dists)) / mean_diff
    #std_dists_ = (np.array(std_dists) - np.min(std_dists)) / std_diff
    #mean_dists_ = np.array(mean_dists) / mean_diff
    #std_dists_ = np.array(std_dists) / std_diff
    #k_opt = ks[ np.argmin( np.mean(mean_dists_+std_dists_) ) ]
    k_opt = ks[ np.argmin(std_dists) ]
    results = {'k_opt': k_opt,
               #'mean_params': [a_mean, b_mean, c_mean, d_mean],
               #'std_params': [a_std, b_std, c_std, d_std],
               'ks': ks,
               'mean_dists': mean_dists, 'std_dists': std_dists,
               #'mean_dists_pred': mean_dists_pred,
               #'std_dists_pred': std_dists_pred
               }

    data.uns[f'k-opt_{groupby}_results'] = results
    if verbose:
        print(f"Added data.uns['k-opt_{groupby}_results']")




