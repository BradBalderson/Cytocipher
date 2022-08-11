"""
Plotting functions for neuron activation analysis.
"""

import numpy as np
from scanpy import AnnData
from .act_helpers import create_flat_df, _marker_map

import matplotlib.pyplot as plt

def ieg_plot(data: AnnData, ieg: str,
             figsize: tuple=(6.48, 4.8), vmin: float=3, vmax: float=7,
             point_scaler: float=1000, point_scale_exp: float=1,
             cmap: str='Reds', show: bool=True):
    """Dot-plot where each row is the perturbation & each column is the cell
        type. Points are coloured by logFC within each cell type between control
        & treated samples for the inidicated IEG of interest. Point size is by
        the percent difference from control cells that express the IEG.
    """

    # Retrieving the information from the AnnData #
    ieg_logfcs = data.uns['ieg_logfcs'][ieg]
    ieg_prop_expr_diff = data.uns['ieg_prop_expr_diffs'][ieg]

    # Rank by logFC #
    ct_sums = ieg_logfcs.values.sum(axis=1)
    ct_order = np.argsort(-ct_sums)
    social_sums = ieg_logfcs.values.sum(axis=0)
    social_order = np.argsort(social_sums)

    ieg_logfcs_ordered = ieg_logfcs.iloc[ct_order, social_order]
    ieg_prop_expr_diff_ordered = ieg_prop_expr_diff.iloc[ct_order, social_order]

    flat_df = create_flat_df(ieg_prop_expr_diff_ordered,
                                 ieg_logfcs_ordered, )

    ax = _marker_map(flat_df['x'], flat_df['y'],
                    flat_df['values'].astype(float),
                    flat_df['color_values'].values.astype(float), cmap=cmap,
                    vmin=vmin, vmax=vmax, figsize=figsize,
                    square_scaler=point_scaler, square_exp=point_scale_exp,
                    )
    if show:
        plt.show()
    else:
        return ax

def ieg_counts(data: AnnData,
               figsize: tuple=(6.48, 4.8), vmin: float=0, vmax: float=None,
               point_scaler: float=1000, point_scale_exp: float=1,
               cmap: str='Reds', show: bool=True):
    """ Plots the counts of significant IEGs per cell type.
    """
    # Retrieving the information from the AnnData #
    ieg_counts = data.uns['ieg_sig_counts']

    # Rank by no. of significant Limma_DE genes #
    ct_sums = ieg_counts.values.sum(axis=1)
    ct_order = np.argsort(-ct_sums)
    social_sums = ieg_counts.values.sum(axis=0)
    social_order = np.argsort(social_sums)

    ieg_counts_ordered = ieg_counts.iloc[ct_order, social_order]

    flat_df = create_flat_df(ieg_counts_ordered, ieg_counts_ordered )

    ax = _marker_map(flat_df['x'], flat_df['y'],
                     flat_df['values'].astype(float),
                     flat_df['color_values'].values.astype(float), cmap=cmap,
                     vmin=vmin, vmax=vmax, figsize=figsize,
                     square_scaler=point_scaler, square_exp=point_scale_exp,
                     )
    if show:
        plt.show()
    else:
        return ax




