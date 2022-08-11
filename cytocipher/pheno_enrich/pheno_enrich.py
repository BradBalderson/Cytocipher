"""
Higher level interface functions for running phenotype enrichment analysis using
IMPC phenotype-gene association data.
"""

import os
import numpy as np
import pandas as pd

from anndata import AnnData

def load_phenogenes():
    """ Loads the phenotype->gene associations from the IMPC database.
    """
    path = os.path.dirname( os.path.realpath(__file__) )
    db_path = f'{path}/../dbs/pheno-geno_summary.txt'
    db = pd.read_csv(db_path, sep='\t', index_col=0)
    db.index = [pheno.replace(' ', "_") for pheno in db.index.values.astype(str)]
    return db

def run(adata: AnnData, phenogenes: pd.DataFrame,
        method: str='NMF-perm', n_comps: int=10,
        min_gene_set_size: int=10, max_gene_set_size: int=3000,
        ):
    return
