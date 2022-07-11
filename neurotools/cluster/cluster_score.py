"""
Functions for measuring quality of clusters.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

from scipy.spatial import distance
from sklearn.preprocessing import minmax_scale

import numba
from numba.typed import List
from numba import njit, prange

def calc_page_enrich_input(data):
    """ Calculates stats necessary to calculate enrichment score.
    """
    full_expr = data.to_df().values

    gene_means = full_expr.mean(axis=0)
    fcs = np.apply_along_axis(np.subtract, 1, full_expr, gene_means)
    mean_fcs = np.apply_along_axis(np.mean, 1, fcs)
    std_fcs = np.apply_along_axis(np.std, 1, fcs)

    return fcs, mean_fcs, std_fcs

def giotto_page_enrich_min(gene_set, var_names, fcs, mean_fcs, std_fcs):
    """ Calculates enrichment scores with most values pre-calculated.
    """
    gene_indices = [np.where(var_names == gene)[0][0] for gene in gene_set]
    set_fcs = np.apply_along_axis(np.mean, 1, fcs[:, gene_indices])

    giotto_scores = ((set_fcs - mean_fcs) * np.sqrt(len(gene_indices)))/std_fcs
    return giotto_scores

def giotto_page_enrich_geneset(data, gene_set, obs_key: str=None):
    """ Re-implementation of giotto page-enrichment score.
    """
    fcs, mean_fcs, std_fcs = calc_page_enrich_input(data)
    giotto_scores = giotto_page_enrich_min(gene_set, data.var_names.values,
                                                         fcs, mean_fcs, std_fcs)
    if type(obs_key)==type(None):
        return giotto_scores
    else:
        data.obs[obs_key] = giotto_scores

def giotto_page_enrich(data: AnnData, groupby: str,
                       var_groups: str='highly_variable',
                       logfc_cutoff: float=0, padj_cutoff: float=.05,
                       n_top: int=5, cluster_marker_key: str=None,
                       rerun_de: bool=True, gene_order='logfc',
                       verbose: bool=True):
    """ Runs Giotto coexpression enrichment score for Limma_DE genes in each cluster.
    """
    n_top = data.shape[1] if type(n_top)==type(None) else n_top

    #### First performing differential expression...
    if type(cluster_marker_key)==type(None):
        if type(var_groups)!=type(None):
            data_sub = data[:,data.var[var_groups]]
            sc.tl.rank_genes_groups(data_sub, groupby=groupby, use_raw=False)
            data.uns['rank_genes_groups'] = data_sub.uns['rank_genes_groups']
        elif rerun_de:
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

    cluster_genes = data.uns[cluster_marker_key]

    ###### Getting the enrichment scores...
    # Precalculations..
    fcs, mean_fcs, std_fcs = calc_page_enrich_input(data)

    cell_scores = np.zeros((data.shape[0], len(cluster_genes)))
    for i, clusteri in enumerate(cluster_genes):
        cluster_scores_ = giotto_page_enrich_min(cluster_genes[clusteri],
                                                    data.var_names, fcs,
                                                    mean_fcs, std_fcs)
        cell_scores[:, i] = cluster_scores_

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
    """Enriches for the genes in the data"""

    expr_bool = expr > 0
    coexpr_counts = expr_bool.sum(axis=1)

    ### Accounting for case where might have only one marker gene !!
    if expr.shape[1] < min_counts:
        min_counts = expr.shape[1]

    ### Must be coexpression of atleast min_count markers!
    nonzero_indices = np.where(coexpr_counts > 0)[0]
    coexpr_indices = np.where(coexpr_counts >= min_counts)[0]
    cell_scores = np.zeros((expr.shape[0]), dtype=np.float64)
    for i in coexpr_indices:
        expr_probs = np.zeros((coexpr_counts[i]))
        cell_nonzero = np.where(expr_bool[i, :])[0]
        for j, genej in enumerate(cell_nonzero):
            expr_probs[j] = len(
                np.where(expr[nonzero_indices, genej] >= expr[i, genej])[0]) / \
                            expr.shape[0]

        cell_scores[i] = np.log2(coexpr_counts[i] / np.prod(expr_probs))

    return cell_scores

@njit(parallel=True)
def get_enrich_scores(full_expr: np.ndarray, all_genes: np.array,
                      cluster_genes_List: np.array,
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
                  min_counts: int = 2, n_cpus: int=1,
                  verbose: bool = True):
    """ NOTE: unlike the giotto function version, this one assumes have already done DE.
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

    full_expr = data[:, all_genes].X.toarray()

    ###### Getting the enrichment scores...
    cell_scores = get_enrich_scores(full_expr, all_genes,
                                    cluster_genes_List, min_counts)

    ###### Adding to AnnData
    cluster_scores = pd.DataFrame(cell_scores, index=data.obs_names,
                                  columns=list(cluster_genes_dict.keys()))
    data.obsm[f'{groupby}_enrich_scores'] = cluster_scores

    if verbose:
        print(f"Added data.obsm['{groupby}_enrich_scores']")

def coexpr_enrich_labelled(data: sc.AnnData, groupby: str, min_counts: int=2,
                                             n_cpus: int=1, verbose: bool=True):
    """ Coexpression enrichment for cell clusters labelled by gene coexpression.
    """
    # NOTE: can't do this because have case of single gene labels which might
    #       have a gene shared between clusters...
    ### Converting cluster names to marker list
    cluster_names = np.unique( data.obs[groupby].values )
    cluster_markers = {}
    for cluster in cluster_names:
        cluster_markers[cluster] = cluster.split('-')

    ### Add to anndata so can calculate coexpression scores.
    cluster_marker_key = f'{groupby}_markers'
    data.uns[cluster_marker_key] = cluster_markers
    if verbose:
        print(f"Added {groupby}_markers.")

    ### Now running coexpression scoring.
    coexpr_enrich(data, groupby, cluster_marker_key=cluster_marker_key,
                                         min_counts=min_counts, verbose=verbose,
                                                                 n_cpus=n_cpus,)

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
    nonzero_indices = np.where(coexpr_counts_all > 0)[0]
    cell_scores = np.zeros((expr.shape[0]), dtype=np.float64)
    for i in coexpr_indices:
        expr_probs = np.zeros(( expr_pos.shape[1] ))
        cell_nonzero = np.where( expr_bool_pos[i, :] )[0]
        for j, genej in enumerate(cell_nonzero):
            expr_level_count = len(np.where(expr_pos[nonzero_indices, genej]
                                                      >= expr_pos[i, genej])[0])
            expr_probs[j] = expr_level_count / expr.shape[0]

        joint_coexpr_prob = np.prod( expr_probs[expr_probs > 0] )
        cell_scores[i] = np.log2(coexpr_counts_pos[i] / joint_coexpr_prob)

    return cell_scores

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
        gene_indices = np.zeros( genes_.shape, dtype=np.int_ )
        for gene_index, gene in enumerate( genes_ ):
            for gene_index2, gene2 in enumerate( all_genes ):
                if gene == gene2:
                    gene_indices[gene_index] = gene_index2

        #### Getting genes shouldn't be in cluster
        diff_indices = np.zeros(genes_diff.shape, dtype=np.int_)
        for gene_index, gene in enumerate( genes_diff ):
            for gene_index2, gene2 in enumerate(all_genes):
                if gene == gene2:
                    diff_indices[gene_index] = gene_index2

        #### Getting indices of which genes are in what cluster.
        clusts = np.unique(clusts_diff)
        negative_indices = np.zeros((len(clusts)), dtype=np.int_)
        if len(clusts) > 0:
            for clusti, clust in enumerate(clusts):
                for clustj in range(len(clusts_diff)):
                    if clusts_diff[clustj]==clust and \
                            clusts_diff[clustj+1]!=clust:
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
                  min_counts: int = 2, n_cpus: int=1,
                  verbose: bool = True):
    """ NOTE: unlike the giotto function version, this one assumes have already done DE.
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

        ### Getting genes which are different if clusters with similar genes
        cluster_diff_full = []
        cluster_diff_clusters = []
        for clusterj in cluster_genes_dict:
            if cluster!=clusterj:
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

def code_enrich_labelled(data: sc.AnnData, groupby: str, min_counts: int=2,
                                             n_cpus: int=1, verbose: bool=True):
    """ Coexpression enrichment for cell clusters labelled by gene coexpression.
    """
    # NOTE: can't do this because have case of single gene labels which might
    #       have a gene shared between clusters...
    ### Converting cluster names to marker list
    cluster_names = np.unique( data.obs[groupby].values )
    cluster_markers = {}
    for cluster in cluster_names:
        cluster_markers[cluster] = cluster.split('-')

    ### Add to anndata so can calculate coexpression scores.
    cluster_marker_key = f'{groupby}_markers'
    data.uns[cluster_marker_key] = cluster_markers
    if verbose:
        print(f"Added {groupby}_markers.")

    ### Now running coexpression scoring.
    code_enrich(data, groupby, cluster_marker_key=cluster_marker_key,
                                         min_counts=min_counts, verbose=verbose,
                                                                 n_cpus=n_cpus,)

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
# TODO could be good to use this in giotto_enrich above...
def get_markers(data: sc.AnnData, groupby: str,
                var_groups: str = None,
                logfc_cutoff: float = 0, padj_cutoff: float = .05,
                t_cutoff: float=3,
                n_top: int = 5, rerun_de: bool = True, gene_order=None,
                pts: bool=False,
                verbose: bool = True):
    """
    Gets the marker genes as a dictionary...
    """

    if rerun_de:
        if type(var_groups) != type(None):
            data_sub = data[:, data.var[var_groups]]
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
            up_rank = np.argsort(-logfcs_rank.values[up_indices, i])[0:n_top]
        else:
            up_rank = up_indices[0:n_top]

        cluster_genes[cluster] = genes_rank.values[up_rank, i]

    data.uns[f'{groupby}_markers'] = cluster_genes
    if verbose:
        print(f"Added data.uns['{groupby}_markers']")

################################################################################
                        # Junk code #
################################################################################
"""
@njit
def code_score(expr: np.ndarray, in_index_end: int, min_counts: int = 2):
    Enriches for the genes in the data, while controlling for genes that
        shouldn't be in the cells.
    

    ### Need to check all places of expression to get expression probablility
    expr_bool = expr > 0
    coexpr_counts_all = expr_bool.sum(axis=1)

    ### Include cells which coexpress genes in positive set
    expr_bool_pos = expr[:, :in_index_end] > 0
    coexpr_counts_pos = expr_bool_pos.sum(axis=1)

    ### Exclude cells which coexpress genes in the negative gene set...
    expr_bool_neg = expr[:, in_index_end:] > 0
    coexpr_counts_neg = expr_bool_neg.sum(axis=1)

    ### Accounting for case where might have only one marker gene !!
    if in_index_end < min_counts:
        min_counts = in_index_end

    ### Getting which cells coexpress atleast min_counts of
    ###  positive set but not negative set
    coexpr_bool = np.logical_and(coexpr_counts_pos >= min_counts, 
                                 coexpr_counts_neg < min_counts)
    coexpr_indices = np.where(coexpr_bool)[0]

    ### Need to check all nonzero indices to get expression level frequency.
    nonzero_indices = np.where(coexpr_counts_all > 0)[0]
    cell_scores = np.zeros((expr.shape[0]), dtype=np.float64)
    for i in coexpr_indices:
        expr_probs = np.zeros(( expr.shape[1] ))
        cell_nonzero = np.where( expr_bool[i, :] )[0]
        for genej in cell_nonzero:
            expr_probs[genej] = len(
                np.where(expr[nonzero_indices, genej] >= expr[i, genej])[0]) / \
                                                                   expr.shape[0]

        # NOTE: if len(diff_indices) is 0, np.prod will return 1.
        out_probs = expr_probs[in_index_end:]
        out_probs = out_probs[out_probs>0]
        in_probs = expr_probs[:in_index_end]
        in_probs = in_probs[in_probs>0]
        cell_scores[i] = np.log2( np.prod(out_probs) / np.prod(in_probs) )

    return cell_scores
"""
