"""
Neurotools automated labelling & merging of clusters based on Limma_DE genes that were used to construct the
nearest neighbour graph for clustering & UMAP construction.

Works by labelling clusters based on Limma_DE genes (user defined criteria), & then
for any cluster with no Limma_DE genes based on this criteria, gets merged with the most
similar cluster to it. Then repeat attempt to label, and so on until all clusters
labelled with top Limma_DE genes from neurotools balanced feature selection.
"""

import numpy as np
import pandas as pd
import scanpy as sc

import scipy.spatial as spatial
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans

import numba
from numba import njit, prange
from numba.typed import List

import scipy.spatial as spatial
from ..utils.general import summarise_data_fast
from ..plotting.utils import add_colors
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


def get_pairs(avg_data: np.ndarray, label_set: np.array):
    """Gets nearest neighbour to each datapoint.
    """
    pairs = []
    dists = []
    point_tree = spatial.cKDTree(avg_data)
    for i, labeli in enumerate(label_set):
        nearest_info = point_tree.query(avg_data[i, :], k=2)
        nearest_dist = nearest_info[0][-1]
        nearest_index = nearest_info[1][-1]
        dists.append(nearest_dist)
        pairs.append([labeli, label_set[nearest_index]])
    pairs = np.array(pairs, dtype='object')
    dists = np.array(dists)
    order = np.argsort(-dists)

    return pairs[order]

def get_pairs_cell(expr: pd.DataFrame, cluster_labels: np.array):
    """ Get nearest neighbour from the expression data, by first averaging.
    """
    label_set = np.unique(cluster_labels)
    avg_data = average(expr, cluster_labels, label_set)
    return get_pairs(avg_data, label_set), label_set

def get_genesubset_de(data: sc.AnnData, data_sub: sc.AnnData,
                      groupby: str, de_key: str):
    """ Gets Limma_DE genes for subset of the data, but adds to the full data.
    """
    sc.tl.rank_genes_groups(data_sub, groupby=groupby, key_added=de_key)
    data.uns[de_key] = data_sub.uns[de_key]

def label_clusters(data: sc.AnnData, groupby: str, de_key: str,
                   reference_genes: np.array,
                   max_genes: int, min_de: int, t_cutoff: float,
                   logfc_cutoff: float, padj_cutoff: float,
                   ref_prefix: bool=True, #Whether to always prefix cluster with reference gene
                   exclude_genes: np.array=None,
                   ):
    """ Labels the clusters based on top Limma_DE genes.
    """
    #### Calling which genes are Limma_DE
    genes_df = pd.DataFrame(data.uns[de_key]['names'])
    tvals_df = pd.DataFrame(data.uns[de_key]['scores'])
    padjs_df = pd.DataFrame(data.uns[de_key]['pvals_adj'])
    logfcs_df = pd.DataFrame(data.uns[de_key]['logfoldchanges'])

    de_bool = tvals_df.values >= t_cutoff
    if padj_cutoff < 1 or logfc_cutoff > 0:  # necessary to add other criteria...
        de_bool = np.logical_and(de_bool, padjs_df.values < padj_cutoff)
        de_bool = np.logical_and(de_bool, logfcs_df.values >= logfc_cutoff)

    ##### Subsetting only to reference genes which were called DE
    if type(reference_genes) != type(None):
        reference_genes = np.array([gene for gene in reference_genes
                                if gene in genes_df.values[:,0]])

    ##### Labelling clusters based on Limma_DE genes
    label_map = {}
    for i, cluster in enumerate(genes_df.columns):
        de_indices = np.where(de_bool[:, i])[0]
        if len(de_indices) >= min_de:  # Significant number of Limma_DE genes detected !!!!
            genes_ = genes_df.values[:, i]
            if type(exclude_genes) != type(None):
                genes_ = np.array([gene for gene in genes_
                                                  if gene not in exclude_genes])
            de_genes = np.unique(genes_[de_indices][0:max_genes])  # Sorts alphabetically

            # Need to put the reference genes first...
            if type(reference_genes) != type(None):
                ref_de = [gene for gene in de_genes if gene in reference_genes]
                other_de = [gene for gene in de_genes if
                            gene not in reference_genes]
                if len(ref_de) == 0 and ref_prefix:  # If no reference genes Limma_DE, then put the first reference gene with max t-value
                    #ref_indices = [np.where(genes_ == ref)[0][0] for ref in
                    #                                            reference_genes]
                    #highest_index = np.argmax(tvals_df.values[ref_indices, i])
                    #ref_de = [reference_genes[highest_index]]
                    ref_de = [gene for gene in genes_
                              if gene in reference_genes][0:1]
                de_genes = ref_de + other_de

            label_map[cluster] = '-'.join(de_genes[0:max_genes])

        else:  # Non-significant number of Limma_DE genes, cluster isn't labelled.
            label_map[cluster] = cluster

    new_labels = np.array(
        [label_map[clust] for clust in data.obs[groupby].values])

    return label_map, new_labels

def merge_neighbours(expr: pd.DataFrame, cluster_labels: np.array,
                     label_pairs: np.array=None):
    """ Merges unlabelled clusters to most similar neighbour.
    """
    ##### Getting the neighbours of each cluster based on average expression
    if type(label_pairs)==type(None):
        label_pairs, label_set = get_pairs_cell(expr, cluster_labels)

    #### Getting groups of clusters which will be merged...
    merge_groups = [] # List of lists, specifying groups of clusters to merge
    for pair in label_pairs:
        added = False
        if pair[0].isdigit(): # Indicates it is an unlabelled cluster.
            for merge_group in merge_groups: #Check if add to existing group
                if np.any([pair_ in merge_group for pair_ in pair]):
                    merge_group.extend(pair)
                    added = True
                    break

            if not added: #Make new group if unmerged group and need to be added
                merge_groups.append(list(pair))

    #### Getting mapping from current clusters to merge clusters
    cluster_map = {}
    for i in range(len(merge_groups)):  # For the merge groups
        for cluster in merge_groups[i]:
            cluster_map[cluster] = str(i)

    for label in label_set:  # Simply map to the same cluster if not merging
        if label not in cluster_map:
            cluster_map[label] = label

    merge_cluster_labels = np.array(
                               [cluster_map[clust] for clust in cluster_labels])

    return cluster_map, merge_cluster_labels

def add_labels(data: sc.AnnData, merge_col: str, cluster_labels: np.array,
               verbose: bool):
    """Adds labels to data"""
    data.obs[merge_col] = cluster_labels
    data.obs[merge_col] = data.obs[merge_col].astype('category')
    if verbose:
        print(f"Added data.obs[ '{merge_col}' ]")

def cluster_label(data: sc.AnnData, var_key: str,
                  groupby: str = 'leiden',
                  obs_key: str = 'cluster_labels',
                  # Stores final cluster labellings!
                  reference_genes: np.array = None,
                  # reference genes used for nts bfs, put these genes first for cluster label
                  max_genes: int = 5, min_de: int = 1, t_cutoff: float = 10,
                  de_key: str = 'rank_genes_groups',
                  logfc_cutoff: float = 0, padj_cutoff: float = 1,
                  iterative_merge: bool=True,
                  verbose: bool = True,
                  ):
    """ Labels clusters by the top Limma_DE genes, and subsequently merges clusters
        with no Limma_DE genes to the most similar cluster.
    """

    if verbose:
        print("Start no. clusters: ",
              len(np.unique(data.obs[groupby].values)))

    #### Getting genes of interest
    genes_bool = data.var[var_key].values

    ### First calling differential expression
    data_sub = data[:, genes_bool]
    get_genesubset_de(data, data_sub, groupby, de_key)

    expr = data_sub.to_df()

    ### Now label clusters based on top Limma_DE genes according to criteria
    label_map, cluster_labels = label_clusters(data, groupby, de_key,
                                               reference_genes,
                                               max_genes, min_de, t_cutoff,
                                               logfc_cutoff, padj_cutoff)

    ### Now performing progressive merging if not all clusters labelled!
    # NOTE: each iteration has two mergings; merge based on labels & based on neighbour merging
    i = 0
    while iterative_merge and np.any([label_map[key].isdigit() for key in
                                   label_map]):  # Still clusters without label.
        i += 1

        #### Adding intermediate clustering results...
        merge_col = f'{groupby}_merge{i}'
        add_labels(data, merge_col, cluster_labels, verbose)

        #### Merging unlabelled clusters to most similar cluster
        label_map, cluster_labels = merge_neighbours(expr, cluster_labels)
        i += 1
        merge_col = f'{groupby}_merge{i}'
        add_labels(data, merge_col, cluster_labels, verbose)

        #### Calling de
        add_labels(data_sub, merge_col, cluster_labels, False)
        get_genesubset_de(data, data_sub, merge_col, de_key)

        #### Relabelling
        label_map, cluster_labels = label_clusters(data, merge_col,
                                                   # Now labelling based on merge_col
                                                   de_key, reference_genes,
                                                   max_genes, min_de, t_cutoff,
                                                   logfc_cutoff, padj_cutoff)

    #### Iteration complete, so can now add final labelled clusters.
    add_labels(data, obs_key, cluster_labels, verbose)
    add_colors(data, obs_key, 'tab20')  # Also adding colors
    if verbose:
        print("Final no. clusters: ", len(np.unique(cluster_labels)))


################################################################################
             # Equivalent to the above approach, except cluster considered #
            # significant if considered differentiable from it's most similar #
            # cluster. #
################################################################################
@njit(parallel=True)
def get_tvals_ranked(expr: np.ndarray, cluster_labels: np.array,
                                              clusts: np.array, refs: np.array):
    """ Getting the Welch's t-vals from predefined comparisons...
    Ref, see ttest_ind_from_stats, unequal variance implementation:
    https://github.com/scipy/scipy/blob/v1.8.1/scipy/stats/_stats_py.py#L5775-L5898
    """
    tvals = np.zeros((expr.shape[1], len(clusts)))
    for i in prange(len(clusts)):
        # Need to count first so we know size of array to make for indices
        clust_n, ref_n = 0, 0
        for iter_ in range(len(cluster_labels)):
            clust_n += 1 if cluster_labels[iter_]==clusts[i] else 0
            ref_n += 1 if cluster_labels[iter_]==refs[i] else 0

        # Getting the relevant indices to subset array
        clust_indices = np.zeros((clust_n), dtype=np.int64)
        ref_indices = np.zeros((ref_n), dtype=np.int64)
        clust_i, ref_i = 0, 0
        for iter_ in range(len(cluster_labels)):
            if cluster_labels[iter_] == clusts[i]:
                clust_indices[clust_i] = iter_
                clust_i += 1
            elif cluster_labels[iter_] == refs[i]:
                ref_indices[ref_i] = iter_
                ref_i += 1

        for j in range(expr.shape[1]): # For each gene
            clust_expr = expr[clust_indices, j]
            ref_expr = expr[ref_indices, j]

            ##### Getting summary stats for tvalue calculation
            mean_clust = np.mean( clust_expr )
            mean_ref = np.mean( ref_expr )
            stderr_clust = np.var( clust_expr ) / clust_n
            stderr_ref = np.var( ref_expr ) / ref_n
            t_ = (mean_clust - mean_ref) / np.sqrt( stderr_clust+stderr_ref )

            tvals[j, i] = t_

    return tvals

def get_nearestcluster_genesubset_de(data: sc.AnnData, data_sub: sc.AnnData,
                                     groupby: str, de_key: str,
                                  expr: pd.DataFrame, cluster_labels: np.array,
                                     scanpy_method: bool=False):
    """ Determines differentially expressed genes between each cluster and
        it's nearest neighbour, adding the results to .uns[de_key] in the same
        format as they normally running sc.tl.rank_genes_groups.
        Also gets what the nearest clusters are!
    """

    ###### Getting the nearest neighbours
    label_pairs, label_set = get_pairs_cell(expr, cluster_labels)

    ##### Adding DE information for each pair.
    de_info = {}
    if scanpy_method:
        gene_dfs, tval_dfs, padj_dfs, logfc_dfs = [], [], [], []
        for pair_ in label_pairs:
            cluster_bool = np.logical_or(cluster_labels == pair_[0],
                                         cluster_labels == pair_[1])
            data_sub_pair = data_sub[cluster_bool, :]
            sc.tl.rank_genes_groups(data_sub_pair, groupby=groupby,
                                    key_added=de_key, groups=[pair_[0]])

            gene_dfs.append( pd.DataFrame(data_sub_pair.uns[de_key]['names']) )
            tval_dfs.append( pd.DataFrame(data_sub_pair.uns[de_key]['scores']) )
            padj_dfs.append( pd.DataFrame(data_sub_pair.uns[de_key][
                                                                 'pvals_adj']) )
            logfc_dfs.append( pd.DataFrame(data_sub_pair.uns[de_key]
                                                           ['logfoldchanges']) )
        #### Concatenating DE information
        de_info['names'] = pd.concat(gene_dfs, axis=1)
        de_info['scores'] = pd.concat(tval_dfs, axis=1)
        de_info['pvals_adj'] = pd.concat(padj_dfs, axis=1)
        de_info['logfoldchanges'] = pd.concat(logfc_dfs, axis=1)
    else:
        clusts, refs = [], []
        max_str = 0
        for pair in label_pairs:
            clusts.append( pair[0] )
            refs.append( pair[1] )
            max_str = max([max_str, len(pair[0])])
        str_dtype = f"<U{max_str}"
        clusts = np.array(clusts, dtype=str_dtype)
        refs = np.array(refs, dtype=str_dtype)

        tvals = get_tvals_ranked(expr.values, cluster_labels, clusts, refs)
        de_info['scores'] = pd.DataFrame(tvals,
                                        index=expr.columns.values.astype(str),
                                         columns=clusts)

    data.uns[de_key] = de_info

    return label_pairs

def merge_neighbours_unlabelled(data: sc.AnnData,
                                expr: pd.DataFrame, cluster_labels: np.array,
                                      de_key: str, min_de: int, t_cutoff: float,
                                        logfc_cutoff: float, padj_cutoff: float,
                                                    label_pairs: np.array=None):
    """ Merges non-significantly different clusters.
    """
    ##### Getting the neighbours of each cluster based on average expression
    if type(label_pairs) == type(None):
        label_pairs, label_set = get_pairs_cell(expr, cluster_labels)

    ##### Retrieving the DE results to decide which clusters to merge.
    # NOTE get absolute values of tvals and logfcs since don't care about
    # direction of difference.
    tvals_df = pd.DataFrame(data.uns[de_key]['scores']).abs() #Absolute value

    de_bool = tvals_df.values >= t_cutoff
    if padj_cutoff < 1 or logfc_cutoff > 0:  # necessary to add other criteria...
        padjs_df = pd.DataFrame(data.uns[de_key]['pvals_adj'])
        logfcs_df = pd.DataFrame(data.uns[de_key]['logfoldchanges']).abs()
        de_bool = np.logical_and(de_bool, padjs_df.values < padj_cutoff)
        de_bool = np.logical_and(de_bool, logfcs_df.values >= logfc_cutoff)

    #### Getting groups of clusters which will be merged...
    merge_groups = []  # List of lists, specifying groups of clusters to merge
    for i, pair in enumerate(label_pairs):
        added = False
        de_indices = np.where(de_bool[:, i])[0]
        if len(de_indices) < min_de:  # Indicates is cluster needs merging!
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

    for pair in label_pairs:  # Simply map to the same cluster if not merging
        if pair[0] not in cluster_map:
            cluster_map[pair[0]] = pair[0]

    merge_cluster_labels = np.array(
                               [cluster_map[clust] for clust in cluster_labels])

    return cluster_map, merge_cluster_labels

def cluster_label_nearest(data: sc.AnnData, var_key: str,
                  groupby: str = 'leiden',
                  obs_key: str = 'cluster_labels',
                  # Stores final cluster labellings!
                  reference_genes: np.array = None,
                  # reference genes used for nts bfs, put these genes first for cluster label
                  max_genes: int = 5, min_de: int = 1, t_cutoff: float = 10,
                  de_key: str = 'rank_genes_groups',
                  logfc_cutoff: float = 0, padj_cutoff: float = 1,
                  iterative_merge: bool=True, n_cpus: int=1,
                  verbose: bool = True,
                  ):
    """ Merges clusters which are not significantly different to their nearest
                                                                      neighbour.
    """
    # Setting threads for paralellisation #
    if type(n_cpus) != type(None):
        numba.set_num_threads(n_cpus)

    if verbose:
        print("Start no. clusters: ",
              len(np.unique(data.obs[groupby].values)))

    #### Getting genes of interest
    genes_bool = data.var[var_key].values

    ### First calling differential expression
    data_sub = data[:, genes_bool]
    expr = data_sub.to_df()
    cluster_labels = data.obs[groupby].values.astype(str)
    label_pairs = get_nearestcluster_genesubset_de(data, data_sub, groupby,
                                                   de_key, expr, cluster_labels)
    if verbose:
        print("Finished DE for clusters versus nearest neighbour.")

    ### Now merge clusters which are not significantly different according to
    ### criteria.
    cluster_map, cluster_labels_new = merge_neighbours_unlabelled(data,
                                                           expr, cluster_labels,
                                                       de_key, min_de, t_cutoff,
                                                      logfc_cutoff, padj_cutoff,
                                                                    label_pairs)

    ### Now performing progressive merging if no convergence!
    # NOTE: each iteration has two mergings; merge based on labels & based on neighbour merging
    i = 0
    while iterative_merge and \
                    np.any(cluster_labels!=cluster_labels_new): # Not converged!
        #### Adding intermediate clustering results...
        merge_col = f'{groupby}_neighbour_merge{i}'
        add_labels(data, merge_col, cluster_labels_new, verbose)

        ### Changing what's considered 'new' and 'old'
        cluster_labels = cluster_labels_new

        #### Calling DE
        add_labels(data_sub, merge_col, cluster_labels, False)
        label_pairs = get_nearestcluster_genesubset_de(data, data_sub,
                                                       merge_col, de_key, expr,
                                                                 cluster_labels)
        if verbose:
            print("Finished DE for clusters versus nearest neighbour.")

        #### Performing a new merge operation.
        cluster_map, cluster_labels_new = merge_neighbours_unlabelled(
                                                     data, expr, cluster_labels,
                                                     de_key, min_de, t_cutoff,
                                                     logfc_cutoff, padj_cutoff,
                                                                    label_pairs)

        i += 1

    #### Iteration complete, so can now add final labelled clusters.
    #### TODO out final cluster labels...
    ####  The final labels will be first-most the DE genes differentiating the
    ####  cluster from everything else, then DE genes that separate cluster
    ####  from nearest neighbour...
    add_labels(data, obs_key, cluster_labels, verbose)
    add_colors(data, obs_key, 'tab20')  # Also adding colors
    if verbose:
        print("Final no. clusters: ", len(np.unique(cluster_labels)))

def merge_clusters_and_label(data: sc.AnnData, var_key: str,
                  groupby: str = 'leiden',
                  obs_key: str = 'cluster_labels',
                  # Stores final cluster labellings!
                  reference_genes: np.array = None,
                  # reference genes used for nts bfs, put these genes first for cluster label
                  max_genes: int = 5, min_de: int = 1, t_cutoff: float = 10,
                  de_key: str = 'rank_genes_groups',
                  logfc_cutoff: float = 0, padj_cutoff: float = 1,
                  iterative_merge: bool=True, n_cpus: int=1,
                  verbose: bool = True,
                  ):
    """ Merges cluster from initial over-clustering, ensuring a minimum no.
        of de genes between clusters and the rest of the cells, and between
        the cluster and it's nearest neighbour.
    """
    cluster_label(data, var_key=var_key,
                     groupby=groupby,
                     obs_key=obs_key,
                     min_de=min_de, max_genes=max_genes, t_cutoff=t_cutoff,
                     verbose=verbose,
                     )
    cluster_label_nearest(data, var_key=var_key,
                     groupby=obs_key,
                     obs_key=obs_key,
                     min_de=min_de, max_genes=max_genes, t_cutoff=t_cutoff,
                     n_cpus=n_cpus,
                     verbose=verbose,
                     )
    # One last final run to get the labels
    cluster_label(data, var_key=var_key,
                  groupby=obs_key,
                  obs_key=obs_key,
                  min_de=min_de, max_genes=max_genes, t_cutoff=t_cutoff,
                  verbose=verbose,
                  )

################################################################################
     # Working on a new method of cluster merging which considers the
     # coexpression scores themselves as a metric for cluster similarity to
     # determine if they significantly different, as opposed to no. of DE genes.
################################################################################
##### Merging the clusters....
def merge_neighbours_v2(cluster_labels: np.array,
                        label_pairs: np.array):
    """ Merges unlabelled clusters to most similar neighbour.
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
                          k: int = 15, knn: int = 5, random_state=20,
                          p_cut: float=.1, verbose: bool = True):
    """ Gets pairs of clusters which are not significantly different from one another based on the enrichment score.
    """
    ### Extracting required information ###
    labels = data.obs[groupby].values
    enrich_scores = data.obsm[f'{groupby}_enrich_scores']
    label_set = enrich_scores.columns.values
    label_scores = [enrich_scores.values[:, i] for i in range(len(label_set))]

    ### Averaging data to get nearest neighbours ###
    if verbose:
        print("Getting nearest neighbours by enrichment scores.")
    avg_data = average(enrich_scores, labels, label_set)

    neighbours = []
    point_tree = spatial.cKDTree(avg_data)
    for i, labeli in enumerate(label_set):
        nearest_info = point_tree.query(avg_data[i, :], k=knn + 1)
        nearest_indexes = nearest_info[1][1:]

        neighbours.append([label_set[index] for index in nearest_indexes])

    # Now going through the MNNs and testing if their cross-scores are significantly different
    if verbose:
        print(
            "Getting pairs of clusters where atleast in one direction not significantly different from one another.")

    kmeans = KMeans(n_clusters=k, random_state=random_state)

    pairs = []
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
    if verbose:
        print(f"Added data.uns['{groupby}_mutualpairs']")

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


def merge_clusters(data: sc.AnnData, groupby: str,
                   k: int = 15, knn: int = 5, n_top_genes: int = 6,
                   p_cut: float=.1,
                   n_cpus: int = 1, random_state=20, max_iter: int = 5,
                   enrich_method: str = 'code',
                   verbose: bool = True):
    """ Merges the clusters following an expectation maximisation approach...
    """

    ### Initial merge ##
    if verbose:
        print("Initial merge.")

    get_markers(data, groupby, n_top=n_top_genes, verbose=False)
    run_enrich(data, groupby, enrich_method, n_cpus)

    old_labels = data.obs[groupby].values.astype(str)
    merge_clusters_single(data, groupby, f'{groupby}_merged',
                          k=k, knn=knn, random_state=random_state,
                          p_cut=p_cut, verbose=False)

    ## Merging per iteration until convergence ##
    for i in range(max_iter):

        # Running marker gene determination #
        get_markers(data, f'{groupby}_merged', n_top=n_top_genes,
                       verbose=False)

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
    get_markers(data, f'{groupby}_merged', n_top=n_top_genes, verbose=False)

    # Running the enrichment scoring #
    run_enrich(data, f'{groupby}_merged', enrich_method, n_cpus)

    if verbose:
        print(f"Added data.obs[f'{groupby}_merged']")
        print(f"Exiting due to reaching max_iter {max_iter}")



