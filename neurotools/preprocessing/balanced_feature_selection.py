"""
Implements balanced feature selection approach; whereby use marker genes for
different cell types to select additional genes in order to perform cluster
on user's variation of interest.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

from .bbknn_functions import bbknn_modified
from .bfs_helpers import pearson_r,  get_selected_dist, get_selected_corr, \
                       add_selected_to_anndata, calc_coexpr_odds, odds_cutoff, \
                      balanced_feature_selection_graph_core, odds_cutoff_core, \
                        get_selected_batch_counts, get_batch_names

def load_nps():
    """ Loads the neuropeptidergic/dopaminergic genes.
    """
    path = os.path.dirname(os.path.realpath(__file__))
    nps = pd.read_csv(path+'/../dbs/NP-DA_markers.txt', header=None).values[:, 0]
    return nps

def load_sex():
    """Loads sex DE genes from development papers"""
    path = os.path.dirname(os.path.realpath(__file__))
    sex_df = pd.read_csv(path + '/../dbs/sex_genes.txt', sep='\t', index_col=0)
    return sex_df

def balanced_feature_select_graph(data: AnnData, reference_genes: np.array,
                                  batch_key=None, n_total: int=500,
                                  pca: bool=False, n_pcs: int=100,
                                  recompute_pca: bool=True,
                                  cache_pca: bool=True,
                                  use_annoy: bool=False,
                                  metric: str='correlation',
                                  approx: bool=True,
                                  initial_neighbours: int=None,
                                  bg_size: int=10000,
                                  padj_method: str='fdr_bh',
                                  padj_cutoff: float=.01,
                                  min_cells: int=3,
                                  verbose: bool=True,
                                  ):
    """ Performs balanced feature selection, except instead of using Pearson
        correlation across all genes (which won't scale well), performs more
        similar to BBKNN except each "batch" is actually a gene, & the
        neighbours we are getting are the neighbouring genes, & when we get the
        closest neighbourhood genes we make sure they aren't already selected.
        NOTE: Settings of use_annoy=False, metric='correlation' produces results
                almost identical to balanced_feature_select_v2.

    Parameters
    ----------
    data: AnnData
        The data object.
    reference_genes: np.array
        Array of gene names contained in data.var_names to bootstrap additional
        genes.
    batch_key: str
        Column in data.obs which indicates batch, if batch is specified, will
        subset to each particular batch & perform bfs separately for each
        batch of cells before concatentating results; this ensures gene
        coexpression are not distored by batch effect.
    n_total: int
        Total no. of genes to select, balancing selected genes across all
        features.
    pca: bool
        Whether to perform PCA on the data first.
    n_pcs: int
        The no. of pcs to compute.
    recompute_pca: bool
        Whether to recompute the pca or not if it's already present.
    use_annoy: bool
        Whether to use annoy approximation for nearest neighbour graph
        construction.
    metric: str
        Which distance metric to use when contstructing graph.
    approx: bool
        Approximate nearest neighbour graph construction, or not?
    initial_neighbours: bool
        Number of nearest neighbours to get per query gene intially, if smaller
        than n_total or None defaults to n_total.
    bg_size: int
        The no. of random background genes to randomly pair with the query
        genes to set cutoff for significance co-expression between genes.
    padj_method: str
        Method used to adjust p-values.
    padj_cutoff: float
        Adjusted p-value cutoff to use for significant genes to select.
    verbose: bool
        Whether to print output to the user.
    Returns
    -------
    grid_data: AnnData
        Equivalent expression data to adata, except values have been summed by
        cells that fall within defined bins.
    """
    if type(initial_neighbours)==type(None) or initial_neighbours < n_total:
        initial_neighbours = n_total

    # Separating the data by batch information, if any!
    if type(batch_key) == type(None) or batch_key not in data.obs.columns:
        datas = [data]
        batch_set = ['all']
        if batch_key not in data.obs.columns:
            print(f"Warning, {batch_key} not in data.obs, batch_key ignored.")
    else:
        batch_labels = data.obs[batch_key].values.astype(str)
        data.obs[batch_key] = data.obs[batch_key].astype('category')
        if verbose:
            print(f"Set data.obs[{batch_key}] to categorical.")
        batch_set = list( data.obs[batch_key].cat.categories )
        datas = [data[batch_labels==batch_name,:] for batch_name in batch_set]


    # Now getting the balanced selected genes #
    for i, datai in enumerate(datas):
        """ NOTE: below we are using the cells subsetted to batch for BFS, but
                    adding the results to the original AnnData. 
        """
        batch_name = batch_set[i] if len(batch_set) != 1 else "X"
        if verbose:
            print(f"Processing batch {batch_name}...")

        ### Need to subset genes to those expressed in batch
        expri = datai.X if type(datai.X)==np.ndarray else datai.X.toarray()
        gene_cell_counts = (expri > 0).sum(axis=0)
        genes_sub = datai.var_names.values.astype(str)[
                                                   gene_cell_counts > min_cells]
        reference_genesi = np.array([ref for ref in reference_genes
                                                           if ref in genes_sub])
        datai = datai[:, genes_sub] # Fewer possible genes now
        if len(reference_genesi) != len(reference_genes):
            print(f"\tFiltered out "
                  f"{len(reference_genes)-len(reference_genesi)} "
                  f"reference genes in batch {batch_name} due to insufficient "
                  f"expression (expressed in less than min_cells={min_cells}).")

        ## Balanced feature selection from query features
        selected, selected_corrs, selected_match, expr_genes, gene_features = \
                  balanced_feature_selection_graph_core(datai, reference_genesi,
                                         n_total=n_total, batch_name=batch_name,
                              pca=pca, n_pcs=n_pcs, recompute_pca=recompute_pca,
                              use_annoy=use_annoy, metric=metric, approx=approx,
                                          initial_neighbours=initial_neighbours,
                                                                verbose=verbose)
        """ Only 18 of 62 DE genes detected in the selection for Batch1..
        If it's correct, pearsonr should be 1- the selected_corrs...
        
        from scipy.stats import pearsonr
        calc1, calc2 = [], []
        for xi in range(len(reference_genesi), len(selected)):
            #print(selected[xi]==selected_match[xi])
            expr = datai[:,[selected[xi], selected_match[xi]]].to_df()
            calc1.append( 1-pearsonr(expr.values[:,0], expr.values[:,1])[0] )
            calc2.append( selected_corrs[xi] )
            
        Appears very accurate... is getting the correct genes.
        
        ###### Getting overlap with DE genes
        import beautifulcells.visualisation.quick_plots as qpl
        de_groups = ['DEFac'+group for group in datai.uns['marker_genes'] 
                              if np.any([ref in datai.uns['marker_genes'][group] 
                                                  for ref in reference_genesi])]
        factors = data.varm['de_df'][de_groups]
        factors_flat = factors.values.ravel()
        qpl.distrib(factors_flat[factors_flat>1])
        de_genes = []
        [de_genes.extend(factors.index.values[factors[col].values>1])
                                                     for col in factors.columns]
        de_genes = np.unique(de_genes)
        detected = [gene for gene in selected if gene in de_genes]
        print(len(de_genes), len(detected), detected)
        
        #### Are the undetected DE genes correlated with DE in this batch?
        de_genes_grouped = [factors.index.values[factors[col].values>1]
                                                     for col in factors.columns]
        undetected = [[gene for gene in de_genesi if gene not in selected]
                        for de_genesi in de_genes_grouped]
        detected = [[gene for gene in de_genesi if gene in selected]
                        for de_genesi in de_genes_grouped]
        corrs = []
        for i, genes_i in enumerate(undetected):
            for genei in genes_i:
                for genej in detected[i]:
                    expr = datai[:,[genei, genej]].to_df().values
                    corrs.append( pearsonr(expr[:,0], expr[:,1])[0] )
        
        # The genes which weren't detected are not AT ALL correlated with the
        #  detected DE genes, which suggests that perhaps these genes are
        #  effected by the batch effect... 
        
                    
        #
        de_genes = datai.var_names.values[datai.var['de_gene'].values]
        detected = [gene for gene in selected if gene in de_genes]
        print(len(de_genes), len(detected), detected)
        """
        # TODO get this so that gene_features in alignment with data.
        #if pca and cache_pca: # Saves recomputation if run with different params
        #    data.varm[f'{batch_name}_pca'] = gene_features

        ## Adding the results to anndata
        add_selected_to_anndata(data, data.var_names.values.astype(str),
                                selected, selected_corrs, selected_match,
                                         batch_name=batch_name, verbose=verbose)

        ## Now getting the odds scores
        if verbose:
            print("\tCalculating odds-score of coexpression between selected "
                  "genes & reference genes..")
        woRef_indices = [[i for i, gene in enumerate(selected)
                                                if gene not in reference_genes]]
        selected_woRef = selected[woRef_indices]
        selected_match_woRef = selected_match[woRef_indices]
        odds_full = calc_coexpr_odds(datai, reference_genesi,
                                     selected_woRef, selected_match_woRef,
                                     return_odds_full=True)
        #### Make the odds score aligned to full AnnData
        odds_aligned = np.zeros((data.shape[1]))
        odds_aligned[[np.where(data.var_names.values==gene)[0][0]
                      for gene in datai.var_names]] = odds_full
        data.varm[f'{batch_name}_bfs_results'][f'{batch_name}_odds'] = \
                                                                    odds_aligned

        ## Now determining the significantly coexpressed genes
        # Using the odds-score & randomisation to determine cutoff
        if verbose:
            print("\tDynamically determining odds-score based cutoff "
                  "using random genes..")
        results_df, bg = odds_cutoff_core(datai, batch_name, reference_genesi,
                                          selected, odds_full,
                                          bg_size=bg_size,
                                          padj_method=padj_method,
                                          padj_cutoff=padj_cutoff)
        ### Adding results to AnnData
        all_results_df = data.varm[f'{batch_name}_bfs_results']
        ## Adding like this will result in NaNs, this will be useful to keep
        ## track of what genes were detectable per batch...
        final_results_df = pd.concat([all_results_df, results_df], axis=1)

        ########### Make sure certain columns have correct typing !!!
        float_cols = [f'{batch_name}_ps', f'{batch_name}_padjs',
                      f'{batch_name}_sig_odds']
        for col_ in float_cols:
            final_results_df[col_] = final_results_df[col_].astype(float)
        final_results_df[f'{batch_name}_sig'] = final_results_df[
                                               f'{batch_name}_sig'].astype(bool)

        ##### Adding to data...
        data.varm[f'{batch_name}_bfs_results'] = final_results_df
        data.uns[f'{batch_name}_bfs_background'] = bg

def update_odds_cutoff(data, batch_name: str = None, padj_cutoff: float = 0.01,
                       verbose: bool = True):
    """ Updates the odds cutoff with different parameters.
    """
    batch_names = get_batch_names(data, batch_name=batch_name, verbose=verbose)

    for batch_name in batch_names:
        results_df = data.varm[f'{batch_name}_bfs_results']
        padjs = results_df[f'{batch_name}_padjs'].values
        sig_bool = padjs < padj_cutoff
        results_df[f'{batch_name}_sig'] = sig_bool
        odds = results_df[f'{batch_name}_odds'].values.astype(float)
        odds[sig_bool == False] = 0
        results_df[f'{batch_name}_sig_odds'] = odds

        if verbose:
            print(f"Updated data.varm['{batch_name}_bfs_results']")

def get_sig_counts_per_batch(data, verbose: bool=False):
    """ Gets the count of the no. of batches the gene was significant for!
    """
    batch_names = get_batch_names(data, verbose=verbose)
    sig_gene_counts, batch_sig_genes, all_genes = \
                             get_selected_batch_counts(data,
                                                       batch_names, verbose)
    groups = list(batch_names)
    sig_genes_grouped = [list(batch_sig_genes[batch]) for batch in
                         batch_sig_genes]

    data.uns['bfs_sig_gene_count_info'] = {'sig_gene_counts': sig_gene_counts,
                                           'batch_sig_genes': batch_sig_genes,
                                           'all_genes': all_genes,
                                           'groups': groups,
                                         'sig_genes_grouped': sig_genes_grouped,
                                           }
    if verbose:
        print("Added data.uns['bfs_sig_gene_count_info']")









###### Old implementations
def balanced_feature_select_v2(data: AnnData, reference_genes: np.array,
                            n_total: int=500, verbose: bool=True):
    """Similar to the development version, except in this case we don't select
    one gene for each reference gene per iteration, instead select all genes
    for reference at once. Is fater than the original, but still not fast
    enough.
    """
    expr_vals = data.X if type(data.X)==np.ndarray else data.X.toarray()
    expr_genes = np.array(data.var_names, dtype=str)

    # Getting locations of the reference & non-reference genes #
    ref_indices = [np.where(expr_genes == ref_gene)[0][0]
                   for ref_gene in reference_genes]
    not_ref_indices = [i for i in list(range(len(expr_genes)))
                       if i not in ref_indices]
    not_ref_genes = expr_genes[not_ref_indices]

    expr_not_ref = expr_vals[:, not_ref_indices]

    # Correlations between ref genes (rows) & non-ref genes (cols)
    corrs = np.zeros((len(ref_indices), len(not_ref_indices)))
    for i, ref_index in enumerate(ref_indices):
        ref_expr = expr_vals[:, ref_index]
        ref_corrs = np.apply_along_axis(pearson_r, 0, expr_not_ref, ref_expr)

        corrs[i, :] = ref_corrs

    ##### Take the best correlated with each ref until n_total selected ########
    selected, selected_corrs, selected_match = get_selected_corr(corrs,
                                                    reference_genes, expr_genes,
                                                         not_ref_genes, n_total)

    # Adding the results to AnnData #
    add_selected_to_anndata(data, expr_genes, selected, selected_corrs,
                            selected_match, verbose)

    # Post-hoc metric of method performance; odds-score
    if verbose:
        print(
            "Calculating odds-score of coexpression between selected genes & reference genes..")
    calc_coexpr_odds(data, verbose=verbose)

    # Using the odds-score & randomisation to determine cutoff
    if verbose:
        print(
            "Dynamically determining odds-score based cutoff using random genes..")
    odds_cutoff(data, verbose=verbose)

def balanced_feature_select_original(data: AnnData, reference_genes: np.array,
                                     n_total: int=500, verbose=bool):
    """ Original implementation from Development paper.
        Too slow for big data.
    """
    expr_vals = data.X if type(data.X) == np.ndarray else data.X.toarray()
    expr_genes = np.array(data.var_names, dtype=str)

    # Getting locations of the reference & non-reference genes #
    ref_indices = [np.where(expr_genes == ref_gene)[0][0]
                   for ref_gene in reference_genes]
    not_ref_indices = [i for i in list(range(len(expr_genes)))
                       if i not in ref_indices]
    not_ref_genes = expr_genes[not_ref_indices]

    expr_not_ref = expr_vals[:, not_ref_indices]

    # Correlations between ref genes (rows) & non-ref genes (cols)
    corrs = np.zeros((len(ref_indices), len(not_ref_indices)))
    for i, ref_index in enumerate(ref_indices):
        ref_expr = expr_vals[:, ref_index]
        ref_corrs = np.apply_along_axis(pearson_r, 0, expr_not_ref, ref_expr)

        corrs[i, :] = ref_corrs

    ##### Take the best correlated with each ref until n_total selected ########
    # Determining max_gene length for making array #
    selected = np.empty((n_total), dtype=expr_genes.dtype)
    for i, ref in enumerate(reference_genes):
        selected[i] = ref
    selected_corrs = np.ones((n_total))
    selected_match = selected.copy()  # Genes the selected match with
    remaining_indices = np.zeros((corrs.shape[1]), dtype=np.int64)
    for i in range(corrs.shape[1]):
        remaining_indices[i] = i
    ref_index = 0
    n_selected = len(reference_genes)
    while n_selected < n_total:
        corrs_sub = corrs[:, remaining_indices]
        values = corrs_sub[ref_index, :]
        order = np.argsort(-values)
        sorted = values[order]
        best_corr, best_i = sorted[0], order[0]
        selected_corrs[n_selected] = best_corr
        selected[n_selected] = not_ref_genes[remaining_indices][best_i]
        selected_match[n_selected] = reference_genes[ref_index]

        index_bool = np.full((len(remaining_indices)), True, dtype=np.bool_)
        index_bool[best_i] = False
        remaining_indices = remaining_indices[index_bool]

        if ref_index == len(reference_genes) - 1:
            ref_index = 0
        else:
            ref_index += 1

        n_selected += 1

        # Adding the results to AnnData #
        add_selected_to_anndata(data, expr_genes, selected, selected_corrs,
                                selected_match, verbose)

        # Post-hoc metric of method performance; odds-score
        if verbose:
            print(
                "Calculating odds-score of coexpression between selected genes & reference genes..")
        calc_coexpr_odds(data, verbose)

""" Junk Code
def bfs_run(data: AnnData, reference_genes: np.array,
            method: str="graphNN", n_total: int=500, verbose: bool=True):
    Key function running the different balanced feature selection methods.
    
    method_to_function = {"graphNN": balanced_feature_select_graph,
                          "v2": balanced_feature_select_v2,
                          "original": balanced_feature_select_original}
    if method not in method_to_function:
        raise Exception(f"Method must be: {list(method_to_function.keys())}")

    bfs_function = method_to_function[method]
    bfs_function(data, reference_genes, n_total=n_total, verbose=verbose)
"""

