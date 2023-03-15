"""
Functions measuring quality of clusters by enrichment scoring on marker genes.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale

import numba
from numba.typed import List
from numba import jit, njit, prange

##### COMMON helper methods
def calc_page_enrich_input(data):
    """ Calculates stats necessary to calculate enrichment score.
    """
    full_expr = data.to_df().values

    gene_means = full_expr.mean(axis=0)
    # fcs = np.apply_along_axis(np.subtract, 1, full_expr, gene_means)
    # mean_fcs = np.apply_along_axis(np.mean, 1, fcs)
    # std_fcs = np.apply_along_axis(np.std, 1, fcs)

    return calc_page_enrich_FAST( full_expr, gene_means )

@njit(parallel=True)
def calc_page_enrich_FAST(full_expr: np.ndarray, gene_means: np.array):
    """ Calculates necessary statistics for Giotto enrichment scoring....
    """

    # gene_means = np.zeros((full_expr.shape[1]), dtype=np.float64)
    # for i in range(len(gene_means)):
    #     gene_means[i] = np.mean( full_expr[:,i] )

    n = full_expr.shape[0]
    fcs = np.zeros((n, len(gene_means)), dtype=np.float64)
    mean_fcs = np.zeros((n), dtype=np.float64)
    std_fcs = np.zeros((n), dtype=np.float64)

    for i in prange(n):
        fcs[i, :] = np.subtract(full_expr[i, :], gene_means)
        mean_fcs[i] = np.mean( fcs[i, :] )
        std_fcs[i] = np.std( fcs[i, :] )

    return fcs, mean_fcs, std_fcs

##### METHODs when want to score just one gene set...
def giotto_page_enrich_min(gene_set, var_names, fcs, mean_fcs, std_fcs):
    """ Calculates enrichment scores with most values pre-calculated.
    """
    gene_indices = [np.where(var_names == gene)[0][0] for gene in gene_set]
    set_fcs = np.apply_along_axis(np.mean, 1, fcs[:, gene_indices])

    giotto_scores = ((set_fcs - mean_fcs) * np.sqrt(len(gene_indices)))/std_fcs
    return giotto_scores

def giotto_page_enrich_geneset(data, gene_set, obs_key: str=None,
                               verbose: bool=True):
    """ Re-implementation of giotto page-enrichment score.
    """
    fcs, mean_fcs, std_fcs = calc_page_enrich_input(data)
    giotto_scores = giotto_page_enrich_min(gene_set, data.var_names.values,
                                                         fcs, mean_fcs, std_fcs)
    if type(obs_key)==type(None):
        return giotto_scores
    else:
        data.obs[obs_key] = giotto_scores
        if verbose:
            print(f"Added data.obs['{obs_key}']")

##### METHODs for cluster scoring...
@njit
def giotto_page_enrich_min_FAST(gene_indices, fcs, mean_fcs, std_fcs):
    """ Calculates enrichment scores with most values pre-calculated.
    """
    set_fcs = np.zeros((len(mean_fcs)), dtype=np.float64)

    geneset_fcs = fcs[:, gene_indices]
    for i in range(len(mean_fcs)):
        set_fcs[i] = np.mean( geneset_fcs[i,:] )

    giotto_scores = ((set_fcs - mean_fcs) * np.sqrt(len(gene_indices)))/std_fcs

    return giotto_scores

### Tried jit but getting errors; njit on the above function reduces run time
### to 3 sec for 248 clusters anyhow, better off optimising elsewhere...
#@jit(parallel=True, #forceobj=True,
#     nopython=False)
#@jit(parallel=True)
def giotto_page_percluster(n_cells: int, cluster_genes: dict,
                           var_names: np.array, fcs: np.array,
                           mean_fcs: np.array, std_fcs: np.array):
    """ Runs Giotto PAGE enrichment per cluster!
    """
    cell_scores = np.zeros((n_cells, len(cluster_genes)), dtype=np.float64)
    cluster_names = list(cluster_genes.keys())
    for i in prange(len(cluster_names)):

        clusteri = cluster_names[i]

        if len(cluster_genes[clusteri])==0:
            raise Exception(f"No marker genes for a cluster detected. "
                             f"Rerun with more relaxed marker gene parameters.")

        gene_indices = np.array([np.where(var_names == gene)[0][0]
                                 for gene in cluster_genes[clusteri]],
                                dtype=np.int64)

        cluster_scores_ = giotto_page_enrich_min_FAST(gene_indices, fcs,
                                                     mean_fcs, std_fcs)
        cell_scores[:, i] = cluster_scores_

    return cell_scores

def giotto_page_enrich(data: AnnData, groupby: str,
                       var_groups: str='highly_variable',
                       logfc_cutoff: float=0, padj_cutoff: float=.05,
                       n_top: int=5, cluster_marker_key: str=None,
                       rerun_de: bool=True, gene_order='logfc',
                       n_cpus: int=1,
                       verbose: bool=True):
    """ Runs Giotto PAGE enrichment for cluster markers. Imporant to note that
        by default this function will automatically determine marker genes,
        as opposed to coexpr_enrich and code_enrich. To disable this,
        specify a value for cluster_marker_key, after running get_markers()
        with the same 'groupby' as input.

        Parameters
        ----------
        data: sc.AnnData
            Single cell RNA-seq anndata, QC'd a preprocessed to log-cpm in
                                                                         data.X.
        groupby: str
            Specifies the clusters to merge, defined in data.obs[groupby]. Must
            be categorical type.
        var_groups: str
            Specifies a column in data.var of type boolean, with True indicating
            the candidate genes to use when determining marker genes per cluster.
            Useful to, for example, remove ribosomal and mitochondrial genes.
            None indicates use all genes in data.var_names as candidates.
        logfc_cutoff: float
            Log-FC above which a gene can be considered a marker when comparing
            a given cluster and all other cells.
        padj_cutoff: float
            Adjusted p-value (Benjamini-Hochberg correction) below which a gene
            can be considered a marker gene.
        n_top: int
            The maximimum no. of marker genes per cluster.
        cluster_marker_key: str
            Key in data.uns, which specifies the marker genes for each cluster
            in data.obs[groupby]. In format where keys are the clusters, and
            values are a list of genes in data.var_names.
        rerun_de: bool
            Whether to rerun the DE analysis, or using existing results in
            data.uns['rank_genes_groups']
        gene_order: str
            By default, gets n_top qualifying genes ranked by log-FC.
            Specifying 't' here will rank by t-value, instead.
        verbose: bool
            Print statements during computation (True) or silent run (False).
        Returns
        --------
            data.obsm[f'{groupby}_enrich_scores']
                Cell by cell type data frame, with the coexpr enrichment scores
                for the values.
    """
    numba.set_num_threads( n_cpus )

    n_top = data.shape[1] if type(n_top)==type(None) else n_top

    #### First performing differential expression...
    if type(cluster_marker_key)==type(None) and \
            f'{groupby}_markers' not in data.uns:
        if type(var_groups)!=type(None) and rerun_de:
            data_sub = data[:,data.var[var_groups]]
            sc.tl.rank_genes_groups(data_sub, groupby=groupby, use_raw=False)
            data.uns['rank_genes_groups'] = data_sub.uns['rank_genes_groups']
        elif rerun_de and 'rank_genes_groups' not in data.uns:
            sc.tl.rank_genes_groups(data, groupby=groupby, use_raw=False)

        #### Getting marker genes for each cluster...
        genes_rank = pd.DataFrame(data.uns['rank_genes_groups']['names'])
        logfcs_rank = pd.DataFrame(data.uns['rank_genes_groups']['logfoldchanges'])
        padjs_rank = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj'])

        up_bool = np.logical_and(logfcs_rank.values > logfc_cutoff,
                                 padjs_rank.values < padj_cutoff)

        cluster_genes = {}
        for i, cluster in enumerate(genes_rank.columns):
            up_indices = np.where(up_bool[:,i])[0]
            if gene_order=='logfc':
                up_rank = np.argsort(-logfcs_rank.values[up_indices, i])[0:n_top]
            else:
                up_rank = up_indices[0:n_top]

            cluster_genes[cluster] = genes_rank.values[up_rank, i]

        data.uns[f'{groupby}_markers'] = cluster_genes
        cluster_marker_key = f'{groupby}_markers'
        if verbose:
            print(f"Added data.uns['{groupby}_markers']")
    elif type(cluster_marker_key)==type(None):
        cluster_marker_key = f'{groupby}_markers'

    cluster_genes = data.uns[cluster_marker_key]

    ###### Getting the enrichment scores...
    # Precalculations..
    fcs, mean_fcs, std_fcs = calc_page_enrich_input(data)

    # cell_scores = np.zeros((data.shape[0], len(cluster_genes)))
    # for i, clusteri in enumerate(cluster_genes):
    #     if len(cluster_genes[clusteri])==0:
    #         raise Exception(f"No marker genes for {clusteri}. "
    #                         f"Rerun with more relaxed marker gene parameters.")
    #
    #     cluster_scores_ = giotto_page_enrich_min(cluster_genes[clusteri],
    #                                                 data.var_names, fcs,
    #                                                 mean_fcs, std_fcs)
    #     cell_scores[:, i] = cluster_scores_
    cell_scores = giotto_page_percluster(data.shape[0], cluster_genes,
                                              data.var_names.values.astype(str),
                                                         fcs, mean_fcs, std_fcs)
    # if not np.any( cell_scores ):
    #     raise Exception(f"No marker genes for a cluster detected. "
    #                     f"Rerun with more relaxed marker gene parameters.")

    ###### Adding to AnnData
    cluster_scores = pd.DataFrame(cell_scores, index=data.obs_names,
                                  columns=list(cluster_genes.keys()))
    data.obsm[f'{groupby}_enrich_scores'] = cluster_scores

    if verbose:
        print(f"Added data.obsm['{groupby}_enrich_scores']")

################################################################################
             # Functions related to Coexpression Score #
################################################################################
@njit
def coexpr_score(expr: np.ndarray, min_counts: int = 2):
    """Enriches for the genes in the data. Now optimised.
    """

    expr_bool = expr > 0
    coexpr_counts = expr_bool.sum(axis=1)

    ### Accounting for case where might have only one marker gene !!
    if expr.shape[1] < min_counts:
        min_counts = expr.shape[1]

    ### Must be coexpression of atleast min_count markers!
    nonzero_indices = np.where(coexpr_counts > 0)[0]
    coexpr_indices = np.where(coexpr_counts >= min_counts)[0]
    expr_nonzero = expr[nonzero_indices, :]
    cell_scores = np.zeros((expr.shape[0]), dtype=np.float64)

    for i in coexpr_indices:
        cell_expr_bool = expr_bool[i, :]
        cell_expr = expr[i, :]

        cells_greater_bool = expr_nonzero[:, cell_expr_bool] >= \
                                                       cell_expr[cell_expr_bool]
        expr_probs = cells_greater_bool.sum( axis=0 ) / expr.shape[0]

        joint_coexpr_prob = np.prod( expr_probs )
        cell_scores[i] = np.log2(coexpr_counts[i] / joint_coexpr_prob)

    return cell_scores

@njit(parallel=True)
def get_enrich_scores(full_expr: np.ndarray, all_genes: np.array,
                      cluster_genes_List: List,
                      min_counts: int,
                      ):
    """ Gets the enrichment of the cluster-specific gene combinations in each
        individual cell.
    """
    cell_scores = np.zeros((full_expr.shape[0], len(cluster_genes_List)))
    for i in prange( len(cluster_genes_List) ):
        genes_ = cluster_genes_List[i]
        gene_indices = np.zeros( genes_.shape, dtype=np.int_ )
        for gene_index, gene in enumerate( genes_ ):
            for gene_index2, gene2 in enumerate( all_genes ):
                if gene == gene2:
                    gene_indices[gene_index] = gene_index2

        cluster_scores_ = coexpr_score(full_expr[:, gene_indices],
                                       min_counts=min_counts)
        cell_scores[:, i] = cluster_scores_

    return cell_scores

def coexpr_enrich(data: sc.AnnData, groupby: str,
                  cluster_marker_key: str = None,
                  n_cpus: int=1, min_counts: int = 2,
                  verbose: bool = True):
    """ Runs coexpr enrichment for cluster markers.
        Assumes have ran get_markers() with the same 'groupby' input.

        Parameters
        ----------
        data: sc.AnnData
            Single cell RNA-seq anndata, QC'd a preprocessed to log-cpm in
                                                                         data.X.
        groupby: str
            Specifies the clusters to merge, defined in data.obs[groupby]. Must
            be categorical type.
        cluster_marker_key: str
            Key in data.uns, which specifies the marker genes for each cluster
            in data.obs[groupby]. In format where keys are the clusters, and
            values are a list of genes in data.var_names.
        min_counts: int
            Controls what's considered a 'small' gene set for winsorisation,
            marker gene lists with len(markers)<=min_counts must have all genes
            coexpressed. While marker gene lists with len(markers)>min_counts
            must have atleast len(markers)-1 genes expressed.
        n_cpus: int
            Number of cpus to use.
        verbose: bool
            Print statements during computation (True) or silent run (False).
        Returns
        --------
            data.obsm[f'{groupby}_enrich_scores']
                Cell by cell type data frame, with the coexpr enrichment scores
                for the values.
    """
    # Setting threads for paralellisation #
    if type(n_cpus) != type(None):
        numba.set_num_threads( n_cpus )

    if type(cluster_marker_key) == type(None):
        cluster_marker_key = f'{groupby}_markers'

    cluster_genes_dict = data.uns[cluster_marker_key]

    # Putting all genes into array for speed.
    all_genes = []
    [all_genes.extend(cluster_genes_dict[cluster])
                                              for cluster in cluster_genes_dict]
    # Getting correct typing
    str_dtype = f"<U{max([len(gene_name) for gene_name in all_genes])}"
    all_genes = np.unique( all_genes ).astype(str_dtype)

    #### Need to convert the markers into a Numba compatible format, easiest is
    #### List of numpy arrays.
    cluster_genes_List = List()
    for cluster in cluster_genes_dict:
        #### Genes stratified by cluster
        cluster_genes = np.array([gene for gene in cluster_genes_dict[cluster]],
                                                                dtype=str_dtype)
        cluster_genes_List.append( cluster_genes )

        if len(cluster_genes)==0:
            raise Exception(f"No marker genes for {cluster}. Relax marker gene "
                            f"parameters in cc.tl.get_markers() or decrease "
                            f"Leiden resolution for inputted clusters.")

    full_expr = data[:, all_genes].X.toarray()

    ###### Getting the enrichment scores...
    cell_scores = get_enrich_scores(full_expr, all_genes,
                                    cluster_genes_List, min_counts)

    ###### Adding to AnnData
    cluster_scores = pd.DataFrame(cell_scores, index=data.obs_names,
                                  columns=list(cluster_genes_dict.keys()))
    # Making sure is the same order as the categories...
    cluster_set_ordered = list(data.obs[groupby].cat.categories)
    cluster_scores = cluster_scores.loc[:, cluster_set_ordered]
    data.obsm[f'{groupby}_enrich_scores'] = cluster_scores

    if verbose:
        print(f"Added data.obsm['{groupby}_enrich_scores']")

################################################################################
     # Coexpression scoring but taking into account other cluster genes #
################################################################################
@njit
def get_min_(total_genes: int, min_counts: int):
    """ Gets the minimum no. of genes which must coexpress.
    """
    # Determining cutoff; if only 1 gene, must express it,
    # if two genes, must coexpress both,
    # if more than two genes, must express all except one gene.
    if total_genes < min_counts:
        min_ = total_genes
    elif total_genes > min_counts: #Was a bug here where had these wrong way around...
        min_ = total_genes - 1
    else:
        min_ = min_counts

    return min_

@njit
def get_neg_cells_bool(expr_bool_neg: np.ndarray, negative_indices: List,
                                                           min_counts: int = 2):
    """ Determines indices of which cells should not be score due to
        coexpressing of genes in the negative set.
    """
    neg_cells_bool = np.zeros( (expr_bool_neg.shape[0]) )
    if len(negative_indices) > 0:
        start_index = 0
        for end_index in negative_indices:
            coexpr_counts = expr_bool_neg[:,
                                      start_index:(end_index+1)].sum(axis=1)

            # Determining cutoff
            min_ = get_min_((end_index-start_index)+1, min_counts)

            coexpr_bool = coexpr_counts > min_
            neg_cells_bool[coexpr_bool] = 1

            start_index = end_index + 1 # Go one position further along.

    return neg_cells_bool

@njit
def code_score_cell(expr: np.ndarray, coexpr_counts_all: np.ndarray,
                    coexpr_indices: np.ndarray, expr_pos: np.ndarray,
                    expr_bool_pos: np.ndarray, coexpr_counts_pos: np.ndarray):
    """ Performs code scoring for each cell in a loop...
    """
    ### Need to check all nonzero indices to get expression level frequency.
    nonzero_indices = np.where(coexpr_counts_all > 0)[0]
    expr_pos_nonzero = expr_pos[nonzero_indices, :]
    cell_scores = np.zeros((expr.shape[0]), dtype=np.float64)

    for i in coexpr_indices:
        cell_expr_pos_bool = expr_bool_pos[i, :]
        cell_expr_pos = expr_pos[i, :]

        cells_greater_bool = expr_pos_nonzero[:, cell_expr_pos_bool] >= \
                                               cell_expr_pos[cell_expr_pos_bool]
        expr_probs = cells_greater_bool.sum( axis=0 ) / expr.shape[0]

        joint_coexpr_prob = np.prod( expr_probs )
        cell_scores[i] = np.log2(coexpr_counts_pos[i] / joint_coexpr_prob)

    return cell_scores

@njit
def code_score(expr: np.ndarray, in_index_end: int,
               negative_indices: List, min_counts: int = 2):
    """Enriches for the genes in the data, while controlling for genes that
        shouldn't be in the cells.
    """
    ### Need to check all places of expression to get expression probablility
    expr_bool = expr > 0
    coexpr_counts_all = expr_bool.sum(axis=1)

    ### Include cells which coexpress genes in positive set
    expr_pos = expr[:, :in_index_end] # Get this for downstream calcs
    expr_bool_pos = expr_bool[:, :in_index_end]
    coexpr_counts_pos = expr_bool_pos.sum(axis=1)

    ### Accounting for case where might have only one marker gene !!
    # Determining cutoff
    min_ = get_min_(in_index_end, min_counts)

    ### Getting cells to exclude, since they coexpress genes in negative set.
    neg_cells_bool = get_neg_cells_bool(expr_bool[:, in_index_end:],
                                        negative_indices, min_counts)

    ### Getting which cells coexpress atleast min_counts of
    ###  positive set but not min_counts of negative set
    coexpr_bool = np.logical_and(coexpr_counts_pos >= min_,
                                 neg_cells_bool==0)
    coexpr_indices = np.where( coexpr_bool )[0]

    ### Need to check all nonzero indices to get expression level frequency.
    cell_scores = code_score_cell(expr, coexpr_counts_all, coexpr_indices,
                                  expr_pos, expr_bool_pos, coexpr_counts_pos)
    return cell_scores

@njit
def get_item_indices(items: List, full_items: np.array):
    """ Gets indices of items in larger array...
    """
    item_indices = np.zeros(items.shape, dtype=np.int_)
    for index, item in enumerate(items):
        for index2, item2 in enumerate(full_items):
            if item == item2:
                item_indices[index] = index2

    return item_indices

@njit(parallel=True)
def get_code_scores(full_expr: np.ndarray, all_genes: np.array,
                      cluster_genes_List: List,
                      cluster_diff_List: List,
                      cluster_diff_cluster_List: List,
                      min_counts: int,
                      ):
    """ Gets the enrichment of the cluster-specific gene combinations in each
        individual cell.
    """
    cell_scores = np.zeros((full_expr.shape[0], len(cluster_genes_List)))
    for i in prange( len(cluster_genes_List) ):
        genes_ = cluster_genes_List[i]
        genes_diff = cluster_diff_List[i]
        clusts_diff = cluster_diff_cluster_List[i]

        #### Getting genes should be in cluster
        gene_indices = get_item_indices(genes_, all_genes)

        #### Getting genes shouldn't be in cluster
        diff_indices = get_item_indices(genes_diff, all_genes)

        #### Getting indices of which genes are in what cluster.
        clusts = np.unique(clusts_diff)
        negative_indices = np.zeros((len(clusts)), dtype=np.int_)
        if len(clusts) > 0:
            for clusti, clust in enumerate(clusts):
                for clustj in range(len(clusts_diff)):
                    ### Added in accounting for the end of the list...
                    ###### Turns out this only happens if
                    if (clusts_diff[clustj]==clust and (clustj+1)==len(clusts_diff)) or \
                       (clusts_diff[clustj]==clust and clusts_diff[clustj+1]!=clust):
                        negative_indices[clusti] = clustj
                        break

        #### Now getting the coexpression scores
        all_indices = np.concatenate((gene_indices, diff_indices))
        cluster_scores_ = code_score(full_expr[:, all_indices],
                                     in_index_end=len(gene_indices),
                                     negative_indices=negative_indices,
                                                          min_counts=min_counts)
        cell_scores[:, i] = cluster_scores_

    return cell_scores

def code_enrich(data: sc.AnnData, groupby: str,
                  cluster_marker_key: str = None,
                  n_cpus: int=1, min_counts: int = 2,
                  squash_exception: bool=True,
                  verbose: bool = True):
    """ Runs code enrichment for cluster markers.
        Assumes have ran get_markers() with the same 'groupby' input.

        Parameters
        ----------
        data: sc.AnnData
            Single cell RNA-seq anndata, QC'd a preprocessed to log-cpm in
                                                                         data.X.
        groupby: str
            Specifies the clusters to merge, defined in data.obs[groupby]. Must
            be categorical type.
        cluster_marker_key: str
            Key in data.uns, which specifies the marker genes for each cluster
            in data.obs[groupby]. In format where keys are the clusters, and
            values are a list of genes in data.var_names.
        min_counts: int
            Controls what's considered a 'small' gene set for winsorisation,
            marker gene lists with len(markers)<=min_counts must have all genes
            coexpressed. While marker gene lists with len(markers)>min_counts
            must have atleast len(markers)-1 genes expressed.
        n_cpus: int
            Number of cpus to use.
        squash_exception: bool
            Whether to ignore the edge-case where there is complete overlap of
            marker genes between two clusters, thus these two clusters will be
            score exactly the same having the same set of marker genes. By
            default is false, prompting the using to relax the marker gene
            parameters so additional genes may differentiate clusters.
        verbose: bool
            Print statements during computation (True) or silent run (False).
        Returns
        --------
            data.obsm[f'{groupby}_enrich_scores']
                Cell by cell type data frame, with the coexpr enrichment scores
                for the values.
    """
    # Setting threads for paralellisation #
    if type(n_cpus) != type(None):
        numba.set_num_threads( n_cpus )

    if type(cluster_marker_key) == type(None):
        cluster_marker_key = f'{groupby}_markers'

    cluster_genes_dict = data.uns[cluster_marker_key]

    # Putting all genes into array for speed.
    all_genes = []
    [all_genes.extend(cluster_genes_dict[cluster])
                                              for cluster in cluster_genes_dict]
    # Getting correct typing
    str_dtype = f"<U{max([len(gene_name) for gene_name in all_genes])}"
    all_genes = np.unique( all_genes ).astype(str_dtype)

    str_dtype_clust = f"<U{max([len(clust) for clust in cluster_genes_dict])}"

    #### Need to convert the markers into a Numba compatible format, easiest is
    #### List of numpy arrays.
    cluster_genes_List = List()
    cluster_diff_List = List() #Genes which the clusters shouldn't express!
    cluster_diff_cluster_List = List()#Genes which these cluster belong to!
    for cluster in cluster_genes_dict:
        #### Genes stratified by cluster
        cluster_genes = np.array([gene for gene in cluster_genes_dict[cluster]],
                                                                dtype=str_dtype)
        cluster_genes_List.append( cluster_genes )

        if len(cluster_genes)==0:
            raise Exception(f"No marker genes for {cluster}. Relax marker gene "
                            f"parameters in cc.tl.get_markers() or decrease "
                            f"Leiden resolution for inputted clusters.")

        ### Getting genes which are different if clusters with similar genes
        cluster_diff_full = []
        cluster_diff_clusters = []
        for clusterj in cluster_genes_dict:
            if cluster!=clusterj:

                ##### Accounting for full overlap!!!!
                #if np.all( np.unique(cluster_genes_dict[clusterj])==\
                #                                     np.unique(cluster_genes) ):
                if set(cluster_genes_dict[clusterj])==set(cluster_genes):
                    error = "Full overlap of + and - gene sets detected " + \
                            f"for {cluster} and {clusterj}; suggested to " + \
                            f"increase number of marker genes for scoring."
                    if not squash_exception:
                        raise Exception(error)
                    else:
                        print(error)

                shared_genes_bool = [gene in cluster_genes_dict[clusterj]
                                                      for gene in cluster_genes]

                # If it's possible to score for this cluster due to
                # shared genes by coexpression scoring, get genes different
                # to remove cells that coexpress these.
                min_ = get_min_(len(cluster_genes), min_counts)
                if sum(shared_genes_bool) >= min_:
                    for gene in cluster_genes_dict[clusterj]:
                        if gene not in cluster_genes:
                            cluster_diff_full.append( gene )
                            cluster_diff_clusters.append( clusterj )

        ##### Adding to the Lists
        cluster_diff_full = np.array(cluster_diff_full, dtype=str_dtype)
        cluster_diff_List.append( cluster_diff_full )

        cluster_diff_clusters = np.array(cluster_diff_clusters,
                                                          dtype=str_dtype_clust)
        cluster_diff_cluster_List.append( cluster_diff_clusters )

    full_expr = data[:, all_genes].X.toarray()

    ###### Getting the enrichment scores...
    cell_scores = get_code_scores(full_expr, all_genes, cluster_genes_List,
                                   cluster_diff_List, cluster_diff_cluster_List,
                                                                     min_counts)

    ###### Adding to AnnData
    cluster_scores = pd.DataFrame(cell_scores, index=data.obs_names,
                                  columns=list(cluster_genes_dict.keys()))
    data.obsm[f'{groupby}_enrich_scores'] = cluster_scores

    if verbose:
        print(f"Added data.obsm['{groupby}_enrich_scores']")

################################################################################
                   # Coexpression specificity score #
################################################################################
""" Looking at how specific the coexpression of the gene is for the group of 
    cells!!! i.e. given the coexpression score for each cluster's genes, 
    how specific is cells cluster genes against all other cluster genes ?
"""
def coexpr_specificity_score(data: sc.AnnData, groupby: str,
                             enrich_key: str=None, verbose=True,
                             broader_expr_adjust: bool=False):
    """ How specific is the cluster enrichment score in cell ?
    """
    if type(enrich_key)==type(None):
        expr_scores_df = data.obsm[f'{groupby}_enrich_scores']
    else:
        expr_scores_df = data.obsm[ enrich_key ]

    #### For each cell, min-max scale across it's scores since cosine
    #### sensitive to scale, ref:
    # https://stats.stackexchange.com/questions/292596/is-feature-normalisation-needed-prior-to-computing-cosine-distance
    ##### First min-max features to reduce effect of this on final score
    expr_scores = minmax_scale(expr_scores_df.values, axis=0)  # per enrich scale
    expr_scores = minmax_scale(expr_scores, axis=1) # per cell scale

    #### Distance to only having score in cluster but no other.
    label_set = expr_scores_df.columns.values.astype(str)
    label_include_indices = []
    if broader_expr_adjust:
        label_genes = [label.split('-') for label in label_set]
        for label_index in range(len(label_set)):
            include_indices = [label_index] + \
                          [i for i in range(len(label_set))
                           if i != label_index and
                           not np.all([gene in label_genes[i]
                                       for gene in label_genes[label_index]])]
            label_include_indices.append( include_indices )

    labels = data.obs[groupby].values.astype(str)

    spec_scores = np.zeros( (data.shape[0]) )
    for celli in range( data.shape[0] ):
        perfect_score = np.zeros( (len(label_set)) )
        label_bool = label_set == labels[celli]
        if sum(label_bool) == 0: # Cell type not scored, so is automatically 0
            continue

        label_index = np.where(label_bool)[0][0]
        perfect_score[label_index] = 1 # Just score for cluster

        # Adjust for cases where clusters express genes in this cluster!!
        if broader_expr_adjust:
            include_indices = label_include_indices[label_index]
        else:
            include_indices = list(range(len(label_set)))

        spec_scores[celli] = 1-distance.cosine(perfect_score[include_indices],
                                             expr_scores[celli, include_indices])

    data.obs[f'{groupby}_specificity'] = spec_scores
    if verbose:
        print(f"Added data.obs['{groupby}_specificity']")

################################################################################
                        # Currently not in use #
################################################################################
def get_markers(data: sc.AnnData, groupby: str,
                var_groups: str = None,
                logfc_cutoff: float = 0, padj_cutoff: float = .05,
                t_cutoff: float=3,
                n_top: int = 5, rerun_de: bool = True, gene_order=None,
                pts: bool=False, min_de: int=0,
                verbose: bool = True):
    """ Gets marker genes per cluster.

    Parameters
        ----------
        data: sc.AnnData
            Single cell RNA-seq anndata, QC'd a preprocessed to log-cpm in
                                                                         data.X.
        groupby: str
            Specifies the clusters to perform one-versus-rest Welch's t-test
            comparison of genes for.
            Must specify defined column in data.obs[groupby].
            Must be categorical type.
        var_groups: str
            Specifies a column in data.var of type boolean, with True indicating
            the candidate genes to use when determining marker genes per cluster.
            Useful to, for example, remove ribosomal and mitochondrial genes.
            None indicates use all genes in data.var_names as candidates.
        logfc_cutoff: float
            Minimum logfc for a gene to be a considered a marker gene for a
            given cluster.
        marker_padj_cutoff: float
            Adjusted p-value (Benjamini-Hochberg correction) below which a gene
            can be considered a marker gene.
        t_cutoff: float
            The minimum t-value a gene must have to be considered a marker gene
            (Welch's t-statistic with one-versus-rest comparison).
        n_top: int
            The maximimum no. of marker genes per cluster.
        rerun_de: bool
            Whether to rerun the DE analysis, or using existing results in
            data.uns['rank_genes_groups']. Useful if have ran get_markers()
            with the same 'groupby' as input, but want to adjust the other
            parameters to determine marker genes.
        gene_order: str
            By default, gets n_top qualifying genes ranked by t-value.
            Specifying logfc here will rank by log-FC, instead.
        pts: bool
            Whether to calculate percentage cells expressing gene within/without
            of each cluster. Only relevant if rerun_de=True.
        min_de: int
            Minimum number of genes to use as markers, if not criteria met.
        verbose: bool
            Print statements during computation (True) or silent run (False).
        Returns
        --------
            data.uns[f'{groupby}_markers']
                Dictionary with cluster names as keys, and list of marker
                genes as values.
    """

    if rerun_de:
        if type(var_groups) != type(None):
            #data_sub = data[:, data.var[var_groups]]

            ## Updating how the data is subsetting so prevents making a deep
            ## copy which can cause memory issues with BIG datasets!!!!
            genes_bool = data.var[var_groups].values

            X_sub = data.X[:, genes_bool]

            data_sub = sc.AnnData(X_sub)
            data_sub.obs[groupby] = data.obs[groupby].values
            data_sub.obs[groupby] = data_sub.obs[groupby].astype('category')
            data_sub.var_names = data.var_names.values[genes_bool]

            sc.tl.rank_genes_groups(data_sub, groupby=groupby, use_raw=False,
                                    pts=pts)
            data.uns['rank_genes_groups'] = data_sub.uns['rank_genes_groups']
        else:
            sc.tl.rank_genes_groups(data, groupby=groupby, use_raw=False,
                                    pts=pts)

    #### Getting marker genes for each cluster...
    genes_rank = pd.DataFrame(data.uns['rank_genes_groups']['names'])
    tvals_rank = pd.DataFrame(data.uns['rank_genes_groups']['scores'])
    logfcs_rank = pd.DataFrame(data.uns['rank_genes_groups']['logfoldchanges'])
    padjs_rank = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj'])

    up_bool = np.logical_and(logfcs_rank.values > logfc_cutoff,
                             padjs_rank.values < padj_cutoff)
    up_bool = np.logical_and(up_bool, tvals_rank.values > t_cutoff)

    cluster_genes = {}
    for i, cluster in enumerate(genes_rank.columns):
        up_indices = np.where(up_bool[:, i])[0]
        if gene_order == 'logfc':
            order = np.argsort(-logfcs_rank.values[up_indices, i])
            up_rank = up_indices[order[0:n_top]]
        else:
            up_rank = up_indices[0:n_top]

        if len(up_indices)==0 and min_de>0:
            up_rank = list(range(min_de))

        cluster_genes[cluster] = genes_rank.values[up_rank, i]

    data.uns[f'{groupby}_markers'] = cluster_genes
    if verbose:
        print(f"Added data.uns['{groupby}_markers']")

################################################################################
 # Methods for normalizing scores and assigning to cell type based on score #
################################################################################
def scale_scores(data: sc.AnnData, enrich_scores_key: str,
                 result_key: str=None, verbose: bool=True):
    """ Minmax scales the enrichment scores, first performed across features
        (i.e. so each gene set score is on scale 0 to 1), then per observation
        (i.e. so then each cell gets a score of 0 to 1 per feature).
    """
    cell_scores_df = data.obsm[enrich_scores_key]

    ##### Handling scale, only min-max implemented.
    expr_scores = cell_scores_df.values
    expr_scores = minmax_scale(expr_scores, axis=0)  # per enrich scale
    expr_scores = minmax_scale(expr_scores, axis=1) # per cell scale
    cell_scores_df = pd.DataFrame(expr_scores, index=cell_scores_df.index,
                                             columns=cell_scores_df.columns)

    if type(result_key)==type(None):
        result_key = f'{enrich_scores_key}_scaled'

    data.obsm[result_key] = cell_scores_df
    if verbose:
        print(f"Added data.obsm['{result_key}']")

def assign_cells(data: sc.AnnData, label_key: str, enrich_scores_key: str,
                 verbose: bool=True):
    """Assigns each cell to the cluster where it has the maximum score.
        If has no scores, then does not get labelled.
    """
    cell_scores_df = data.obsm[enrich_scores_key]

    max_vals = np.apply_along_axis(np.max, 1, cell_scores_df.values)
    max_indices = np.apply_along_axis(np.argmax, 1, cell_scores_df.values)
    np_labels = np.array(
        [cell_scores_df.columns.values[index] for index in max_indices])
    np_labels[max_vals == 0] = ''
    data.obs[label_key] = np_labels
    data.obs[label_key] = data.obs[label_key].astype('category')

    if verbose:
        print(f"Added data.obs['{label_key}']")
