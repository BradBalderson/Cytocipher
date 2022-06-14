""" Wrapper for performing the LR GO analysis.
"""

import os
from ..utils.r_helpers import rpy2_setup, ro, localconverter, pandas2ri

def run_GO(genes, bg_genes, species, r_path, p_cutoff=0.01, q_cutoff=0.5, onts="BP"):
    """Running GO term analysis."""

    # Setting up the R environment #
    rpy2_setup(r_path)

    # Adding the source R code #
    r = ro.r
    path = os.path.dirname(os.path.realpath(__file__))
    r["source"](path + "/go.R")

    # Loading the GO analysis function #
    go_analyse_r = ro.globalenv["GO_analyse"]

    # Running the function on the genes #
    genes_r = ro.StrVector(genes)
    if type(bg_genes) != type(None):
        bg_genes_r = ro.StrVector(bg_genes)
    else:
        bg_genes_r = ro.r["as.null"]()
    p_cutoff_r = ro.FloatVector([p_cutoff])
    q_cutoff_r = ro.FloatVector([q_cutoff])
    onts_r = ro.StrVector([onts])
    go_results_r = go_analyse_r(
        genes_r, bg_genes_r, species, p_cutoff_r, q_cutoff_r, onts_r
    )
    with localconverter(ro.default_converter + pandas2ri.converter):
        go_results = ro.conversion.rpy2py(go_results_r)

    return go_results
