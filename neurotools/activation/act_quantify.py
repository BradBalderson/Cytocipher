"""
Functions for quantifying neuronal activation levels in scRNA-seq data between
treated & control data.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

def load_iegs():
    """Loading the intermediate-early genes.
    """
    path = os.path.dirname(os.path.realpath(__file__))
    genes_df = pd.read_csv(path+'/../dbs/Kim_TableS2_IEGs.csv', sep='\t',
                           index_col=0)
    iegs = genes_df['IEGs'].values.astype(str)
    iegs = iegs[iegs!='nan']

    return iegs

def ieg_activation(data: AnnData, iegs: np.array,
                   celltype_col: str, perturb_col: str, perturb_control: str,
                   logfc_cutoff: float=2, padj_cutoff: float=.05,
                   verbose: bool=True):
    """ Quantifies neuronal activate via intermediate-early gene activity within
        cell types between two different conditions. Experimental design is
        ACT-seq.
    """

    #### Splitting up the datas into cell types ####
    socials = np.unique(data.obs[perturb_col].values)
    ct_labels = data.obs[celltype_col].values.astype(str)
    ct_set = np.unique(ct_labels)

    iegs_ = [gene for gene in iegs if gene in data.var_names]
    if verbose:
        missing = [gene for gene in iegs if gene not in data.var_names]
        print(f"Warning: missing some iegs; {missing}")

    datas = [data[ct_labels == ct, iegs_].copy() for ct in ct_set]

    #### Calling de iegs between behaviour & control #####
    ct_ieg_stats = {}
    ieg_sigs = pd.DataFrame(index=ct_set, columns=socials[socials!='Control'],
                            dtype=float)
    ieg_logfcs = {}
    ieg_prop_exprs = {}
    ieg_prop_expr_diffs = {}
    for ieg in iegs_:
        ieg_logfcs[ieg] = pd.DataFrame(index=ct_set, columns=socials,
                                       dtype=float)
        ieg_prop_exprs[ieg] = pd.DataFrame(index=ct_set, columns=socials,
                                           dtype=float)
        ieg_prop_expr_diffs[ieg] = pd.DataFrame(index=ct_set, columns=socials,
                                                dtype=float)

    for i, data_ct in enumerate(datas):
        ct = ct_set[i]
        sc.tl.rank_genes_groups(data_ct, groupby=perturb_col, pts=True,
                                reference=perturb_control, use_raw=False,
                                )
        genes_df = pd.DataFrame(data_ct.uns['rank_genes_groups']['names'],
                                dtype=str)
        logfcs_df = pd.DataFrame(
                             data_ct.uns['rank_genes_groups']['logfoldchanges'],
                                dtype=float)
        pvals_df = pd.DataFrame(data_ct.uns['rank_genes_groups']['pvals'])
        padjs_df = pd.DataFrame(data_ct.uns['rank_genes_groups']['pvals_adj'],
                                dtype=float)
        prop_expr_df = pd.DataFrame(data_ct.uns['rank_genes_groups']['pts'],
                                    dtype=float)

        ### Adding the IEG Expression information for fast plotting ###
        col_indices = list(range(logfcs_df.shape[1]))
        for ieg in iegs_:
            ### Adding in individual IEG information ###
            ieg_logfc = ieg_logfcs[ieg]
            ieg_prop_expr = ieg_prop_exprs[ieg]
            ieg_prop_expr_diff = ieg_prop_expr_diffs[ieg]

            ieg_indices = [np.where(genes_df.values[:, i] == ieg)[0][0]
                           for i in col_indices]
            ieg_logfc.loc[ct, logfcs_df.columns] = logfcs_df.values[ieg_indices,
                                                                     col_indices]
            social_props = prop_expr_df.loc[ieg, :].values
            social_diffs = social_props - prop_expr_df.loc[ieg, perturb_control]
            ieg_prop_expr.loc[ct, prop_expr_df.columns] = social_props
            ieg_prop_expr_diff.loc[ct, prop_expr_df.columns] = social_diffs

        ### Getting the ieg information ###
        fc_bool = logfcs_df.values > logfc_cutoff
        padj_bool = padjs_df.values < padj_cutoff
        sig_bool = np.logical_and(fc_bool, padj_bool)

        ### Counts significant IEGs per cell type ###
        order = [np.where(logfcs_df.columns==social)[0][0]
                                                 for social in ieg_sigs.columns]
        ieg_sigs.loc[ct,:] = sig_bool.sum(axis=1)[order]

        ### Saving overall IEG information per cell type ###
        ieg_info = pd.DataFrame(index=logfcs_df.columns,
                                columns=['IEG_counts', 'IEGs', 'logfcs',
                                       '-log10(padjs)'])
        ieg_info['IEG_counts'] = (sig_bool).sum(axis=0)
        ieg_info['IEGs'] = ['_'.join(genes_df.values[sig_bool[:, i], i])
                                   for i in range(genes_df.shape[1])]
        ieg_info['logfcs'] = ['_'.join(logfcs_df.values[sig_bool[:, i],
                                                               i].astype(str))
                                     for i in range(logfcs_df.shape[1])]
        ieg_info['-log10(padjs)'] = ['_'.join((-np.log10(padjs_df.values[
                                                                    sig_bool[:,
                                                           i], i])).astype(str))
                                            for i in range(padjs_df.shape[1])]


        ieg_stats = {'ieg_info': ieg_info,
                     'genes_ranked': genes_df, 'logfcs': logfcs_df,
                     'pvals': pvals_df, 'padjs': padjs_df, 'sig_bool': sig_bool,
                     'prop_expr': prop_expr_df}
        ct_ieg_stats[ct] = ieg_stats

    for ieg in iegs_:
        ieg_logfcs[ieg] = ieg_logfcs[ieg].drop(columns=perturb_control)
        ieg_prop_expr_diffs[ieg] = ieg_prop_expr_diffs[ieg].drop(
                                                        columns=perturb_control)

    #### Attaching the results to the AnnData object ######
    data.uns['ieg_stats'] = ieg_stats
    data.uns['ieg_info'] = ieg_info
    data.uns['ieg_logfcs'] = ieg_logfcs
    data.uns['ieg_prop_exprs'] = ieg_prop_exprs
    data.uns['ieg_prop_expr_diffs'] = ieg_prop_expr_diffs
    data.uns['ieg_sig_counts'] = ieg_sigs

    if verbose:
        print("Added data.uns['ieg_stats']")
        print("Added data.uns['ieg_info']")
        print("Added data.uns['ieg_logfcs']")
        print("Added data.uns['ieg_prop_exprs']")
        print("Added data.uns['ieg_prop_expr_diffs']")
        print("Added data.uns['ieg_sig_counts']")






