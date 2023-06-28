""" Key functions for Cytocipher cluster merge functionality.
"""

import numpy as np
import pandas as pd
import scanpy as sc


from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from sklearn.cluster import KMeans

from collections import defaultdict
from numba.typed import List
from numba import jit

from .cluster_score import giotto_page_enrich, code_enrich, coexpr_enrich, \
                                                                     get_markers
from ._group_methods import group_scores
from ._neighbors import enrich_neighbours, all_neighbours, general_neighbours
from ..utils.general import summarise_data_fast

def average(expr: pd.DataFrame, labels: np.array, label_set: np.array):
    """Averages the expression by label.
    """
    label_indices = List()
    [label_indices.append(np.where(labels == label)[0]) for label in label_set]
    if type(expr) == pd.DataFrame:
        expr = expr.values
    avg_data = summarise_data_fast(expr, label_indices)

    return avg_data

def get_merge_groups_SLOW(label_pairs: list):
    """Examines the pairs to be merged, and groups them into large groups of
        of clusters to be merged. This implementation will merge cluster pairs
        if there exists a mutual cluster they are both non-significantly
        different from. Can be mediated by filtering the pairs based on the
        overlap of clusters they are both non-significantly different from
        (which is performed in a separate function).
    """
    #### Using a syncing strategy with a dictionary.
    clust_groups = defaultdict(list)
    all_match_bool = [False] * len(label_pairs)
    for pairi, pair in enumerate(label_pairs):
        # NOTE we only need to do it for one clust of pair,
        # since below syncs for other clust
        clust_groups[pair[0]].extend(pair)
        clust_groups[pair[0]] = list(np.unique(clust_groups[pair[0]]))

        # Pull in the clusts from each other clust.
        for clust in clust_groups[pair[0]]:  # Syncing across clusters.
            clust_groups[pair[0]].extend(clust_groups[clust])
            clust_groups[pair[0]] = list(np.unique(clust_groups[pair[0]]))

        # Update each other clust with this clusters clusts to merge
        for clust in clust_groups[pair[0]]:  # Syncing across clusters.
            clust_groups[clust].extend(clust_groups[pair[0]])
            clust_groups[clust] = list(np.unique(clust_groups[clust]))

        # Checking to make sure they now all represent the same thing....
        clusts = clust_groups[pair[0]]
        match_bool = [False] * len(clusts)
        for i, clust in enumerate(clusts):
            match_bool[i] = np.all(
                np.array(clust_groups[clust]) == np.array(clusts))

        all_match_bool[pairi] = np.all(match_bool)

    # Just for testing purposes...
    #print(np.all(all_match_bool))

    # Getting the merge groups now.
    merge_groups = []
    merge_groups_str = []
    all_groups = list(clust_groups.values())
    for group in all_groups:
        group_str = '_'.join(group)
        if group_str not in merge_groups_str:
            merge_groups.append( group )
            merge_groups_str.append( group_str )

    return merge_groups

@jit(parallel=False, forceobj=True, nopython=False)
def get_merge_groups(label_pairs: list):
    """Examines the pairs to be merged, and groups them into large groups of
        of clusters to be merged. This implementation will merge cluster pairs
        if there exists a mutual cluster they are both non-significantly
        different from. Can be mediated by filtering the pairs based on the
        overlap of clusters they are both non-significantly different from
        (which is performed in a separate function).
    """
    #### Using a syncing strategy with a dictionary.
    clust_groups = defaultdict(set)
    #all_match_bool = [False] * len(label_pairs)
    for pairi, pair in enumerate(label_pairs):
        # NOTE we only need to do it for one clust of pair,
        # since below syncs for other clust
        clust_groups[pair[0]] = clust_groups[pair[0]].union(pair)

        # Pull in the clusts from each other clust.
        for clust in clust_groups[pair[0]]:  # Syncing across clusters.
            clust_groups[pair[0]] = clust_groups[pair[0]].union(
                                                           clust_groups[clust] )

        # Update each other clust with this clusters clusts to merge
        for clust in clust_groups[pair[0]]:  # Syncing across clusters.
            clust_groups[clust] = clust_groups[clust].union(
                                                         clust_groups[pair[0]] )

        # Checking to make sure they now all represent the same thing....
        # clusts = clust_groups[pair[0]]
        # match_bool = [False] * len(clusts)
        # for i, clust in enumerate(clusts):
        #     match_bool[i] = np.all(
        #         np.array(clust_groups[clust]) == np.array(clusts))
        #
        # all_match_bool[pairi] = np.all(match_bool)

    # Just for testing purposes...
    #print(np.all(all_match_bool))

    # Getting the merge groups now.
    merge_groups = [] #np.unique([tuple(group) for group in clust_groups.values()])
    merge_groups_str = []
    all_groups = list(clust_groups.values())
    for group in all_groups:
        group = list(group)
        group.sort()
        group_str = '_'.join(group)
        if group_str not in merge_groups_str:
            merge_groups.append( group )
            merge_groups_str.append( group_str )

    return merge_groups

##### Merging the clusters....
def merge_neighbours_v2(cluster_labels: np.array,
                        label_pairs: list):
    """ Merges pairs of clusters specified in label pairs, giving a new set
        of labels per cell with the clusters merged, as well as a dictionary
        mapping the original cluster label to the merged cluster labels.
    """
    label_set = np.unique(cluster_labels)

    #### Getting groups of clusters which will be merged...
    merge_groups = get_merge_groups( label_pairs )

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

def filter_pairs(label_pairs: list, nonsig_overlap_cutoff: float):
    """ Filters the pairs to be merged based on the proportion of overlapping
        non-significantly different clusters each pair has in common. This is
        used to mediate the degree of cluster merging due to clusters
        having one or more mutual clusters from which they are non-significantly,
        but the two clusters themselves are signficantly different. Setting
        to nonsig_overlap_cutoff to 1 would mean two clusters would need to have
        all of the same nonsig clusters to be merged. 0.5 would mean half
        overlapping, etc.
    """
    # First, for each cluster, get the set of other clusters it is not
    # significantly different from.
    clust_groups = defaultdict(list)
    for pairi, pair in enumerate(label_pairs):
        clust_groups[pair[0]].extend(pair)
        clust_groups[pair[0]] = list(np.unique(clust_groups[pair[0]]))

        clust_groups[pair[1]].extend(pair)
        clust_groups[pair[1]] = list(np.unique(clust_groups[pair[1]]))

    # Second, get overlaps proportion between two pairs of nonsig clusters
    merge_clusts = np.array(list(clust_groups.keys()))
    overlap_props = np.zeros((len(merge_clusts), len(merge_clusts)))
    filt_merge_pairs = []
    for i, clust in enumerate(merge_clusts):
        clust_clusts = clust_groups[clust]
        for merge_clust in clust_clusts:

            j = np.where(merge_clusts == merge_clust)[0][0]
            if j >= i:  # Is identical across diagonal.
                continue

            merge_clust_clusts = clust_groups[merge_clust]
            overlap = [clust_ for clust_ in clust_clusts if
                       clust_ in merge_clust_clusts]
            total = len(np.unique(clust_clusts + merge_clust_clusts))

            overlap_props[i, j] = len(overlap) / total
            overlap_props[j, i] = overlap_props[i, j]

            if overlap_props[i, j] > nonsig_overlap_cutoff:
                filt_merge_pairs.append((clust, merge_clust))

    overlap_props = pd.DataFrame(overlap_props, index=merge_clusts,
                                 columns=merge_clusts)

    return filt_merge_pairs, overlap_props

def merge(data: sc.AnnData, groupby: str,
          p_cut: float, key_added: str=None,
          use_p_adjust: bool=True,

          nonsig_overlap_cutoff: float=0,
          verbose: bool=True):
    """ Updates the merged clusters using internally stored p-values in data
        with different merge parameters.

        Parameters
        ----------
        data: AnnData
            Single cell data on which cc.tl.merge_clusters has been performed.
        groupby: str
            The overclusters which were merged.
        p_cut: float
            The p-value cutoff to use to perform merging again.
        use_p_adjust: bool
            Whether to use the adjusted p-values (FDR adjusted).
        nonsig_overlap_cutoff: float
            This deals with the case where clusters which ARE significantly
            different can be merged IF they are mutually non-significantly
            different from a different cluster. This value indicates the
            proportion of the same clusters two clusters must BOTH be
            non-significantly different from to be merged. Takes value from 0 to
            1; with 0 indicating to merge clusters if they have ANY cluster
            they are both mutually non-significnalty different from, and 1
            indicating clusters must have the SAME set of clusters they are
            mutually significantly different from for them to be merged.
        verbose: bool
            Write operation details to standard out?
    """
    if type(key_added)==type(None):
        key_added = f'{groupby}_merged'

    if use_p_adjust:
        pvals_dict = data.uns[f'{groupby}_padjs']
    else:
        pvals_dict = data.uns[f'{groupby}_ps']

    pvals = np.array( list(pvals_dict.values()) )
    pairs = np.array( list(pvals_dict.keys()) )

    nonsig_pairs_ = pairs[pvals >= p_cut]
    # Pair names processed so not sensitive to '_' in input names
    #nonsig_pairs = [tuple(pair.split('_')) for pair in nonsig_pairs]
    clusters = np.unique( data.obs[groupby].values )

    nonsig_pairs = []
    for clust1 in clusters:
        for clust2 in clusters:
            if f'{clust1}_{clust2}' in nonsig_pairs_:
                nonsig_pairs.append( (clust2, clust1) )

    merge_pairs = []
    for pairi in nonsig_pairs:
        for pairj in nonsig_pairs:
            if pairi[0] == pairj[1] and pairi[1] == pairj[0] \
                    and pairi not in merge_pairs and pairj not in merge_pairs:

                merge_pairs.append( pairi )

    # Can filter the merge_pairs by how many overlapping nonsig clusters they have
    if nonsig_overlap_cutoff > 0 : #If it's zero, same as not filtering...
        merge_pairs, _ = filter_pairs(merge_pairs, nonsig_overlap_cutoff)

    #### Now merging..
    labels = data.obs[groupby].values.astype(str)
    cluster_map, merge_cluster_labels = merge_neighbours_v2(labels,
                                                            merge_pairs)

    data.obs[key_added] = merge_cluster_labels
    data.obs[key_added] = data.obs[key_added].astype('category')
    data.obs[key_added] = data.obs[key_added].cat.set_categories(
                        np.unique(merge_cluster_labels.astype(int)).astype(str))

    #### Recording the merge pairs
    data.uns[f'{groupby}_mutualpairs'] = merge_pairs

    if verbose:
        print(f"Added data.uns['{groupby}_mutualpairs']")
        print(f"Added data.obs['{key_added}']")

##### Getting MNNs based on the scores
def merge_clusters_single(data: sc.AnnData, groupby: str, key_added: str,
                          k: int = 15,
                          #knn: int = None,
                          mnn_frac_cutoff: float = None,
                          random_state: int=20,
                          p_cut: float=.1, score_group_method: str='kmeans',
                          p_adjust: bool=False, p_adjust_method: str='fdr_bh',
                          verbose: bool = True):
    """ Gets pairs of clusters which are not significantly different from one
        another based on the enrichment score.
    """
    ### Extracting required information ###
    labels = data.obs[groupby].values
    enrich_scores = data.obsm[f'{groupby}_enrich_scores']
    label_set = enrich_scores.columns.values
    label_scores = [enrich_scores.values[:, i] for i in range(len(label_set))]

    """ ### From OLD version, limited to MNNs via enrich scores.
    if type(knn)!=type(None) and knn < (len(label_set)-1):
        ### MNNs via euclidean distance on avg enrich scores ###
        neighbours, dists = enrich_neighbours(enrich_scores, labels, label_set,
                                              knn, verbose)
    """
    #### General cluster neighbours, determined by prior user-run neighbours.
    if type(mnn_frac_cutoff)!=type(None):
        neighbours, dists, clust_dists = general_neighbours(data, labels,
                                                     label_set, mnn_frac_cutoff)

    else:
        ### Pairwise cluster comparison default ###
        neighbours, dists = all_neighbours(label_set)

    data.uns[f'{groupby}_neighbours'] = {label: neighbours[i]
                                           for i, label in enumerate(label_set)}
    data.uns[f'{groupby}_neighdists'] = {label: dists[i]
                                           for i, label in enumerate(label_set)}

    # Now going through the MNNs and testing if their cross-scores are significantly different
    if verbose:
        print(
            "Getting pairs of clusters where atleast in one direction not significantly different from one another.")

    if score_group_method=='kmeans':
        kmeans = KMeans(n_clusters=k, random_state=random_state)
    else:
        kmeans = None

    ps_dict = {}
    for i, labeli in enumerate(label_set):
        for j, labelj in enumerate(label_set):
            ### Only compare mutual neighbours ###
            if labelj in neighbours[i] and labeli in neighbours[j]:

                labeli_labelj_scores = label_scores[j][labels == labeli]
                labelj_labelj_scores = label_scores[j][labels == labelj]

                ##### Performing the aggregation of scores that correct for
                ##### bias toward pairs with higher abundance.
                labeli_labelj_scores_mean = group_scores(labeli_labelj_scores,
                                                         score_group_method, k,
                                                         kmeans)
                labelj_labelj_scores_mean = group_scores(labelj_labelj_scores,
                                                         score_group_method, k,
                                                         kmeans)

                t, p = ttest_ind(labeli_labelj_scores_mean,
                                 labelj_labelj_scores_mean)

                #### Above outputs nan if all 0's for one-case;
                #### Indicates significant difference in case where one or the
                #### other cluster have all zero's, but if both have all zeros
                #### then non-significant difference!!!!
                if np.isnan(p):
                    n_all_zero = sum([
                               np.all(np.array(labeli_labelj_scores_mean) == 0),
                               np.all(np.array(labelj_labelj_scores_mean) == 0)
                                     ])
                    if n_all_zero == 1: #One is all zero but other isn't, sig!
                        p = 0
                    else: #Both are all zero, non-sig!!
                        p = 1

                ps_dict[f'{labeli}_{labelj}'] = p

    #### Adding p-values and adjusted p-values
    data.uns[f'{groupby}_ps'] = ps_dict

    pvals = np.array(list(ps_dict.values()))
    pairs = np.array(list(ps_dict.keys()))
    padjs = multipletests(pvals, method=p_adjust_method)[1]
    data.uns[f'{groupby}_padjs'] = {pair: padjs[i] for i, pair in
                                                               enumerate(pairs)}

    #### Performing the actual merge operation based on the p-values.
    #### Function can be run after p-value calculations if wish to adjust
    #### these later.
    merge(data, groupby, p_cut, key_added=key_added, use_p_adjust=p_adjust,
                                                                verbose=verbose)

    # Now identifying pairs which are mutually not-significant from one another;
    # i.e. cluster 1 is not signicant from cluster 2, and cluster 2 not significant from cluster 1.
    if verbose:
        print(f"Added data.uns['{groupby}_ps']")
        print(f"Added data.uns['{groupby}_padjs']")
        print(f"Added data.uns['{groupby}_neighbours']")
        print(f"Added data.uns['{groupby}_neighdists']")
        print("Getting pairs of clusters which are mutually not different from "
              "one another.")

def run_enrich(data: sc.AnnData, groupby: str, enrich_method: str,
               n_cpus: int, squash_exception: bool=False):
    """ Runs desired enrichment method.
    """
    enrich_options = ['code', 'coexpr', 'giotto']
    if enrich_method not in enrich_options:
        raise Exception(
       f"Got enrich_method={enrich_method}; expected one of : {enrich_options}")

    if enrich_method == 'code':
        code_enrich(data, groupby, n_cpus=n_cpus, verbose=False,
                    squash_exception=squash_exception)
    elif enrich_method == 'coexpr':
        coexpr_enrich(data, groupby, n_cpus=n_cpus, verbose=False)
    elif enrich_method == 'giotto':
        giotto_page_enrich(data, groupby,
                              rerun_de=False, verbose=False)

# The key function #
def merge_clusters(data: sc.AnnData, groupby: str,
                   var_groups: str=None, n_top_genes: int = 6, t_cutoff: int=3,
                   marker_padj_cutoff: float=.05, gene_order: str=None,
                   min_de: int=0,
                   enrich_method: str = 'code', p_cut: float=0.01,
                   max_iter: int = 0, #knn: int = None,
                   mnn_frac_cutoff: float = None,
                   k: int = 15, random_state=20,
                   n_cpus: int = 1,
                   score_group_method: str='quantiles',
                   p_adjust: bool=True, p_adjust_method: str='fdr_bh',
                   squash_exception: bool=False,
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
    gene_order: str
        Statistic to rank top genes per cluster by; None is t-value, 'logfc'
        indicates to rank by log-FC when taking the top N de genes per cluster.
    min_de: int
        Minimum no. of marker genes to use for each cluster.
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
    mnn_frac_cutoff: float
        The proportion of cells between two clusters which are mutual
        nearest-neighbors for the clusters to be compared. For this to work,
        sc.pp.neighbors(data) must be run first, and can be run on different
        representations of the data by specifying the use_rep parameter in
        sc.pp.neighbors.
    k: int
        k for the k-means clustering of the enrichment scores prior to
        significance testing, to reduce group imbalance bias and inflated
        statistical power due to pseudoreplication.
        Set to None to use each cell as an observation.
    random_state: int
        Random seed for the k-means clustering. Set by default to ensure
        reproducibility each time function run with same data & input params.
    n_cpus: int
        Number of cpus to perform for the computation.
    score_group_method: str
        One of 'kmeans', 'quantile_bin', & 'quantiles'. Determines how the
        scores are aggregated before significance testing; reduces p-value
        inflation & bias of significance toward larger clusters.
    p_adjust: bool
        True to adjust p-values, False otherwise.
    p_adjust_method: str
        Method to use for p-value adjustment. Options are defined by
        statsmodels.stats.multitest.multipletests.
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
        print( "Initial merge." )

    get_markers(data, groupby, n_top=n_top_genes, verbose=False,
                var_groups=var_groups, t_cutoff=t_cutoff,
                padj_cutoff=marker_padj_cutoff,
                gene_order=gene_order, min_de=min_de)
    run_enrich(data, groupby, enrich_method, n_cpus,
               squash_exception=squash_exception)

    old_labels = data.obs[groupby].values.astype(str)

    merge_clusters_single(data, groupby, f'{groupby}_merged',
                          k=k, mnn_frac_cutoff=mnn_frac_cutoff, random_state=random_state,
                          p_cut=p_cut,
                          score_group_method=score_group_method,
                          p_adjust=p_adjust, p_adjust_method=p_adjust_method,
                          verbose=False)

    ## Merging per iteration until convergence ##
    for i in range(max_iter):

        # Running marker gene determination #
        get_markers(data, f'{groupby}_merged', n_top=n_top_genes,
                       verbose=False, var_groups=var_groups, t_cutoff=t_cutoff,
                        padj_cutoff=marker_padj_cutoff,
                        gene_order=gene_order, min_de=min_de)

        # Running the enrichment scoring #
        run_enrich(data, f'{groupby}_merged', enrich_method, n_cpus,
                   squash_exception=squash_exception)

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
                              k=k, mnn_frac_cutoff=mnn_frac_cutoff,
                              random_state=random_state,
                              score_group_method = score_group_method,
                              p_adjust = p_adjust,
                              p_adjust_method = p_adjust_method,
                              p_cut=p_cut, verbose=False)

    ## Reached max iter, exit with current solution ##
    # Running marker gene determination #
    get_markers(data, f'{groupby}_merged', n_top=n_top_genes, verbose=False,
                var_groups=var_groups, t_cutoff=t_cutoff,
                padj_cutoff=marker_padj_cutoff,
                gene_order=gene_order, min_de=min_de)

    # Running the enrichment scoring #
    run_enrich(data, f'{groupby}_merged', enrich_method, n_cpus,
               squash_exception=squash_exception)

    if verbose:
        print(f"Added data.obs[f'{groupby}_merged']")
        print(f"Exiting due to reaching max_iter {max_iter}")

