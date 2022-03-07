""" Helper function for diagnostics plots of neuron activation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_flat_df(int_df, color_df):
    """Reformats a dataframe representing interactions to a flat format."""
    n_rows = int_df.shape[0] * int_df.shape[1]
    flat_df = pd.DataFrame(index=list(range(n_rows)),
                           columns=["x", "y", "values", "color_values"])
    row_i = 0
    for i, indexi in enumerate(int_df.index.values):
        for j, colj in enumerate(int_df.columns.values):
            flat_df.iloc[row_i, :] = [indexi, colj,
                                     int_df.values[i, j], color_df.values[i, j]]
            row_i += 1

    return flat_df


def _marker_map(x, y, size, color_vals, marker='o',
                ylabel='', vmin=None, vmax=None,
             ax=None, figsize=(6.48, 4.8), cmap=None,
                square_scaler=700, square_exp=1):
    """Main underlying helper function for generating the heatmaps."""
    if type(cmap) == type(None):
        cmap = "Spectral_r"

    if type(vmin)==type(None):
        vmin = min(color_vals)
    if type(vmax)==type(None):
        vmax = max(color_vals)

    if type(ax) == type(None):
        fig, ax = plt.subplots(figsize=figsize)

    # Mapping from column names to integer coordinates
    x_labels = list(x.values)  # [v for v in sorted(x.unique())]
    y_labels = list(y.values)  # [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    s = (size / sum(size) * square_scaler)**square_exp
    out = ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s = s,
        c=color_vals,
        cmap=cmap,
        # Vector of square sizes, proportional to size parameter
        marker=marker,  # Shape of markers
        vmin=vmin, vmax=vmax,
    )
    out.set_array(color_vals)
    out.set_clim(vmin, vmax)
    cbar = plt.colorbar(out)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(ylabel, rotation=270)

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment="right")
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    return ax






