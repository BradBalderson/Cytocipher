"""
Plots related to balanced feature selection.
"""

import numpy as np
from scanpy import AnnData

import seaborn as sb
import matplotlib.pyplot as plt

from .general_plots import distrib

########################### Diagnostic plots ###################################
def odds_cutoff(data: AnnData, batch_name: str=None,
                bins: int=100, show: bool=True):
    """Plots distribution of odds-scores & the cutoff for what's considered
        a significant score.
    """
    if type(batch_name)==type(None):
        batch_name = 'X'
    reference_bool = data.varm[f'{batch_name}_bfs_results'][
                                           f'{batch_name}_bfs_reference'].values
    selected_bool = data.varm[f'{batch_name}_bfs_results'][
                                            f'{batch_name}_bfs_selected'].values
    selected_woRef_bool = np.logical_and(selected_bool, reference_bool == False)
    odds = data.varm[f'{batch_name}_bfs_results']['odds'].values[
                                                            selected_woRef_bool]
    sig_odds = data.varm[f'{batch_name}_bfs_results'][
                                                f'{batch_name}_sig_odds'].values
    odds_cutoff = min(sig_odds[sig_odds > 0])

    out = distrib(odds, cutoff=odds_cutoff,
                  x_label=f'Odds-score for {batch_name}', bins=bins, show=show)
    if not show:
        return out

def odds_bg_comp(data: AnnData, batch_name: str=None, show: bool=True):
    """Violin plots which compare the odds scores for
        significant selected genes, selected genes,
        & with random background genes.
    """
    if type(batch_name)==type(None):
        batch_name = 'X'
    rand_odds = data.uns[f'{batch_name}_bfs_background'][
                                               f'{batch_name}_rand_odds'].values
    reference_bool = data.varm[f'{batch_name}_bfs_results'][
                                           f'{batch_name}_bfs_reference'].values
    selected_bool = data.varm[f'{batch_name}_bfs_results'][
                                            f'{batch_name}_bfs_selected'].values
    selected_woRef_bool = np.logical_and(selected_bool, reference_bool == False)
    odds = data.varm[f'{batch_name}_bfs_results'][f'{batch_name}_odds'].values[
                                                            selected_woRef_bool]
    sig_odds = data.varm[f'{batch_name}_bfs_results'][
                                                f'{batch_name}_sig_odds'].values
    sig_odds = sig_odds[np.logical_and(sig_odds > 0, reference_bool == False)]

    odds_list = [sig_odds, odds, rand_odds]
    odds_array = np.array(odds_list, dtype='object').transpose()
    ax = sb.violinplot(data=odds_array,
                       inner='point', cut=0, scale='width')
    # Removing boxes outside #
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Odds-score")
    ax.set_xticklabels(['Significant', 'Foreground', 'Background'])
    if show:
        plt.show()
    else:
        return ax







