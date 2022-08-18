""" Key functions for Cytocipher cluster merge functionality.
"""

import numpy as np
import pandas as pd
import scanpy as sc

import scipy.spatial as spatial
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans

from numba.typed import List

from ..utils.general import summarise_data_fast
from .cluster_score import giotto_page_enrich, code_enrich, coexpr_enrich, \
                                                                     get_markers

def average(expr: pd.DataFrame, labels: np.array, label_set: np.array):
    """Averages the expression by label.
    """
    label_indices = List()
    [label_indices.append(np.where(labels == label)[0]) for label in label_set]
    if type(expr) == pd.DataFrame:
        expr = expr.values
    avg_data = summarise_data_fast(expr, label_indices)

    return avg_data

##### Merging the clusters....
def merge_neighbours_v2(cluster_labels: np.array,
                        label_pairs: np.array):
    """ Merges pairs of clusters specified in label pairs, giving a new set
        of labels per cell with the clusters merged, as well as a dictionary
        mapping the original cluster label to the merged cluster labels.
    """
    ##### Getting the neighbours of each cluster based on average expression
    label_set = np.unique(cluster_labels)

    #### Getting groups of clusters which will be merged...
    merge_groups = []  # List of lists, specifying groups of clusters to merge
    for pair in label_pairs:
        added = False

        for merge_group in merge_groups:  # Check if add to existing group
            if np.any([pair_ in merge_group for pair_ in pair]):
                merge_group.extend(pair)
                added = True
                break

        if not added:  # Make new group if unmerged group and need to be added
            merge_groups.append(list(pair))

    #### Getting mapping from current clusters to merge clusters
    cluster_map = {}
    for i in range(len(merge_groups)):  # For the merge groups
        for cluster in merge_groups[i]:
            cluster_map[cluster] = str(i)

    clusti = len(merge_groups)  # New start of the cluster....
    for label in label_set:
        if label not in cluster_map:
            cluster_map[label] = str(clusti)
            clusti += 1

    merge_cluster_labels = np.array(
                               [cluster_map[clust] for clust in cluster_labels])

    return cluster_map, merge_cluster_labels


##### Getting MNNs based on the scores
def merge_clusters_single(data: sc.AnnData, groupby: str, key_added: str,
                          k: int = 15, knn: int = None, random_state=20,
                          p_cut: float=.1, verbose: bool = True):
    """ Gets pairs of clusters which are not significantly different from one
        another based on the enrichment score.
    """
    ### Extracting required information ###
    labels = data.obs[groupby].values
    enrich_scores = data.obsm[f'{groupby}_enrich_scores']
    label_set = enrich_scores.columns.values
    label_scores = [enrich_scores.values[:, i] for i in range(len(label_set))]

    ### Averaging data to get nearest neighbours ###
    neighbours = []
    dists = []
    if type(knn)!=type(None) and knn < (len(label_set)-1):
        if verbose:
            print("Getting nearest neighbours by enrichment scores.")

        avg_data = average(enrich_scores, labels, label_set)
        point_tree = spatial.cKDTree(avg_data)
        for i, labeli in enumerate(label_set):
            nearest_info = point_tree.query(avg_data[i, :], k=knn + 1)
            nearest_indexes = nearest_info[1]
            dists_ = nearest_info[0]

            neighbours.append([label_set[index] for index in nearest_indexes
                               if label_set[index]!=labeli])
            dists.append( [dist for i_, dist in enumerate(dists_)
                           if label_set[nearest_indexes[i_]]!=labeli] )
    else:
        for label in label_set:
            neighbours.append( list(label_set[label_set!=label]) )
            dists.append( [np.nan]*(len(label_set)-1) )

    data.uns[f'{groupby}_neighbours'] = {label: neighbours[i]
                                         for i, label in enumerate(label_set)}
    data.uns[f'{groupby}_neighdists'] = {label: dists[i]
                                         for i, label in enumerate(label_set)}

    # Now going through the MNNs and testing if their cross-scores are significantly different
    if verbose:
        print(
            "Getting pairs of clusters where atleast in one direction not significantly different from one another.")

    kmeans = KMeans(n_clusters=k, random_state=random_state)

    pairs = []
    ps_dict = {}
    for i, labeli in enumerate(label_set):
        for j, labelj in enumerate(label_set):
            if labelj in neighbours[i] and labeli in neighbours[j]:

                labeli_labelj_scores = label_scores[j][labels == labeli]
                labelj_labelj_scores = label_scores[j][labels == labelj]

                ### Account for n cells <= k
                if len(labeli_labelj_scores) > k:
                    groupsi = kmeans.fit_predict(
                        labeli_labelj_scores.reshape(-1, 1))
                    labeli_labelj_scores_mean = \
                        [np.mean(labeli_labelj_scores[groupsi == k]) for k in
                         np.unique(groupsi)]
                else:
                    labeli_labelj_scores_mean = labeli_labelj_scores

                if len(labelj_labelj_scores) > k:
                    groupsj = kmeans.fit_predict(
                        labelj_labelj_scores.reshape(-1, 1))
                    labelj_labelj_scores_mean = \
                        [np.mean(labelj_labelj_scores[groupsj == k]) for k in
                         np.unique(groupsj)]
                else:
                    labelj_labelj_scores_mean = labelj_labelj_scores

                t, p = ttest_ind(labeli_labelj_scores_mean,
                                 labelj_labelj_scores_mean)
                #### Above outputs nan if all 0's for one-case, indicate significant difference
                if np.isnan(p) and (
                        np.all(np.array(labeli_labelj_scores_mean) == 0) or
                        np.all(np.array(labelj_labelj_scores_mean) == 0)):

                    p = 0

                ps_dict[f'{labeli}_{labelj}'] = p
                if p > p_cut:
                    pairs.append((labeli, labelj))

    # Now identifying pairs which are mutually not-significant from one another;
    # i.e. cluster 1 is not signicant from cluster 2, and cluster 2 not significant from cluster 1.
    if verbose:
        print(
            "Getting pairs of clusters which are mutually not different from one another.")

    mutual_pairs = []
    for pairi in pairs:
        for pairj in pairs:
            if pairi[0] == pairj[1] and pairi[1] == pairj[0] \
                    and pairi not in mutual_pairs and pairj not in mutual_pairs:
                mutual_pairs.append(pairi)

    data.uns[f'{groupby}_mutualpairs'] = mutual_pairs
    data.uns[f'{groupby}_ps'] = ps_dict
    if verbose:
        print(f"Added data.uns['{groupby}_mutualpairs']")
        print(f"Added data.uns['{groupby}_ps']")
        print(f"Added data.uns['{groupby}_neighbours']")
        print(f"Added data.uns['{groupby}_neighdists']")

    # Now merging the non-signficant clusters #
    cluster_map, merge_cluster_labels = merge_neighbours_v2(labels,
                                                               mutual_pairs, )
    data.obs[key_added] = merge_cluster_labels
    data.obs[key_added] = data.obs[key_added].astype('category')
    data.obs[key_added] = data.obs[key_added].cat.set_categories(
        np.unique(merge_cluster_labels.astype(int)).astype(str))

    if verbose:
        print(f"Added data.obs['{key_added}']")


def run_enrich(data: sc.AnnData, groupby: str, enrich_method: str,
               n_cpus: int):
    """ Runs desired enrichment method.
    """
    enrich_options = ['code', 'coexpr', 'giotto']
    if enrich_method not in enrich_options:
        raise Exception(
       f"Got enrich_method={enrich_method}; expected one of : {enrich_options}")

    if enrich_method == 'code':
        code_enrich(data, groupby, n_cpus=n_cpus, verbose=False)
    elif enrich_method == 'coexpr':
        coexpr_enrich(data, groupby, n_cpus=n_cpus, verbose=False)
    elif enrich_method == 'giotto':
        giotto_page_enrich(data, groupby,
                              rerun_de=False, verbose=False)

# The key function #
def merge_clusters(data: sc.AnnData, groupby: str,
                   var_groups: str=None, n_top_genes: int = 6, t_cutoff: int=3,
                   marker_padj_cutoff: float=.05,
                   enrich_method: str = 'code', p_cut: float=.1,
                   max_iter: int = 0, knn: int = None,
                   k: int = 15, random_state=20,
                   n_cpus: int = 1,
                   verbose: bool = True):
    """ Merges the clusters following an expectation maximisation approach.

    Parameters
    ----------
    data: sc.AnnData
        Single cell RNA-seq anndata, QC'd a preprocessed to log-cpm in data.X
    groupby: str
        Specifies the clusters to merge, defined in data.obs[groupby]. Must
        be categorical type.
    var_groups: str
        Specifies a column in data.var of type boolean, with True indicating
        the candidate genes to use when determining marker genes per cluster.
        Useful to, for example, remove ribosomal and mitochondrial genes.
        None indicates use all genes in data.var_names as candidates.
    n_top_genes: int
        The maximimum no. of marker genes per cluster.
    t_cutoff: float
        The minimum t-value a gene must have to be considered a marker gene
        (Welch's t-statistic with one-versus-rest comparison).
    marker_padj_cutoff: float
        Adjusted p-value (Benjamini-Hochberg correction) below which a gene
        can be considered a marker gene.
    enrich_method: str
        Enrichment method to use for scoring cluster membership.
        Must be one of 'code', 'coexpr', or 'giotto'.
    p_cut: float
        P-value cutoff for merging clusters. Cluster pairs above this value are
        considered non-significant and thus merged.
    max_iter: int
        Maximum number of iterations of the expectation-maximisation to perform,
        returns solution at this number of iterations or when convergence
        achieved.
    knn: int
        Number of nearest-neighbours for each cluster to determine, after
        which mutual nearest neighbour clusters are compared. Default None
        indicates to perform pair-wise comparison between clusters.
    k: int
        k for the k-means clustering of the enrichment scores prior to
        significance testing, to reduce inflated statistical power and
        group imbalance.
    random_state: int
        Random seed for the k-means clustering. Set by default to ensure
        reproducibility each time function run with same data & input params.
    n_cpus: int
        Number of cpus to perform for the computation.
    verbose: bool
        Print statements during computation (True) or silent run (False).
    Returns
    --------
        data.obs[f'{groupby}_merged']
            New cell type labels with non-sigificant clusters merged.
        data.uns[f'{groupby}_merged_markers']
            Dictionary with merged cluster names as keys, and list of marker
            genes as values.
        data.obsm[f'{groupby}_merged_enrich_scores']
            Dataframe with cells as rows and merged clusters as columns.
            Values are the enrichment scores for each cluster, using the
            marker genes in data.uns[f'{groupby}_merged_markers']
    """

    ### Initial merge ##
    if verbose:
        print("Initial merge.")

    get_markers(data, groupby, n_top=n_top_genes, verbose=False,
                var_groups=var_groups, t_cutoff=t_cutoff,
                padj_cutoff=marker_padj_cutoff,)
    run_enrich(data, groupby, enrich_method, n_cpus)

    old_labels = data.obs[groupby].values.astype(str)

    merge_clusters_single(data, groupby, f'{groupby}_merged',
                          k=k, knn=knn, random_state=random_state,
                          p_cut=p_cut, verbose=False)

    ## Merging per iteration until convergence ##
    for i in range(max_iter):

        # Running marker gene determination #
        get_markers(data, f'{groupby}_merged', n_top=n_top_genes,
                       verbose=False, var_groups=var_groups, t_cutoff=t_cutoff,
                        padj_cutoff=marker_padj_cutoff,)

        # Running the enrichment scoring #
        run_enrich(data, f'{groupby}_merged', enrich_method, n_cpus)

        # Checking if we have converged #
        new_labels = data.obs[f'{groupby}_merged'].values.astype(str)
        if len(np.unique(old_labels)) == len(np.unique(new_labels)):
            if verbose:
                print(f"Added data.obs[f'{groupby}_merged']")
                print("Exiting due to convergence.")
                return

        if verbose:
            print(f"Merge iteration {i}.")

        # Running new merge operation #
        old_labels = data.obs[f'{groupby}_merged'].values.astype(str)
        merge_clusters_single(data, f'{groupby}_merged', f'{groupby}_merged',
                              k=k, knn=knn, random_state=random_state,
                              p_cut=p_cut, verbose=False)

        #sc.pl.umap(data, color=f'{groupby}_merged')

    ## Reached max iter, exit with current solution ##
    # Running marker gene determination #
    get_markers(data, f'{groupby}_merged', n_top=n_top_genes, verbose=False,
                var_groups=var_groups, t_cutoff=t_cutoff,
                padj_cutoff=marker_padj_cutoff,)

    # Running the enrichment scoring #
    run_enrich(data, f'{groupby}_merged', enrich_method, n_cpus)

    if verbose:
        print(f"Added data.obs[f'{groupby}_merged']")
        print(f"Exiting due to reaching max_iter {max_iter}")

