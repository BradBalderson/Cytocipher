"""
Modified scanorama functions; altered to enable input of which datasets to
integrate with.
"""

import sys
import numpy as np
import pandas as pd
import scanpy as sc

import scanorama
from scanorama import vstack

################################################################################
                   # Modified code from scanorama #
################################################################################
BATCH_SIZE = scanorama.BATCH_SIZE
VERBOSE = scanorama.VERBOSE
DIMRED = scanorama.DIMRED
APPROX = scanorama.APPROX
SIGMA = scanorama.SIGMA
ALPHA = scanorama.ALPHA
KNN = scanorama.KNN

# Integration with scanpy's AnnData object.
def integrate_scanpy(adatas, **kwargs):
    """Integrate a list of `scanpy.api.AnnData`.
    Parameters
    ----------
    adatas : `list` of `scanpy.api.AnnData`
        Data sets to integrate.
    kwargs : `dict`
        See documentation for the `integrate()` method for a full list of
        parameters to use for batch correction.
    Returns
    -------
    None
    """
    datasets_dimred, genes = integrate(
        [adata.X for adata in adatas],
        [adata.var_names.values for adata in adatas],
        **kwargs
    )

    for adata, X_dimred in zip(adatas, datasets_dimred):
        adata.obsm['X_scanorama'] = X_dimred


# Integrate a list of data sets.
def integrate(datasets_full, genes_list, batch_size=BATCH_SIZE,
              verbose=VERBOSE, ds_names=None, dimred=DIMRED, approx=APPROX,
              sigma=SIGMA, alpha=ALPHA, knn=KNN, union=False, hvg=None, seed=0,
              sketch=False, sketch_method='geosketch', sketch_max=10000,
              # Following arguments are added by me to enable adding in information
              # for which datasets to integrate !!!!!
              alignments=None,
              ):
    """Integrate a list of data sets.
    Parameters
    ----------
    datasets_full : `list` of `scipy.sparse.csr_matrix` or of `numpy.ndarray`
        Data sets to integrate and correct.
    genes_list: `list` of `list` of `string`
        List of genes for each data set.
    batch_size: `int`, optional (default: `5000`)
        The batch size used in the alignment vector computation. Useful when
        correcting very large (>100k samples) data sets. Set to large value
        that runs within available memory.
    verbose: `bool` or `int`, optional (default: 2)
        When `True` or not equal to 0, prints logging output.
    ds_names: `list` of `string`, optional
        When `verbose=True`, reports data set names in logging output.
    dimred: `int`, optional (default: 100)
        Dimensionality of integrated embedding.
    approx: `bool`, optional (default: `True`)
        Use approximate nearest neighbors, greatly speeds up matching runtime.
    sigma: `float`, optional (default: 15)
        Correction smoothing parameter on Gaussian kernel.
    alpha: `float`, optional (default: 0.10)
        Alignment score minimum cutoff.
    knn: `int`, optional (default: 20)
        Number of nearest neighbors to use for matching.
    hvg: `int`, optional (default: None)
        Use this number of top highly variable genes based on dispersion.
    seed: `int`, optional (default: 0)
        Random seed to use.
    sketch: `bool`, optional (default: False)
        Apply sketching-based acceleration by first downsampling the datasets.
        See Hie et al., Cell Systems (2019).
    sketch_method: {'geosketch', 'uniform'}, optional (default: `geosketch`)
        Apply the given sketching method to the data. Only used if
        `sketch=True`.
    sketch_max: `int`, optional (default: 10000)
        If a dataset has more cells than `sketch_max`, downsample to
        `sketch_max` using the method provided in `sketch_method`. Only used
        if `sketch=True`.
    Returns
    -------
    integrated, genes
        Returns a two-tuple containing a list of `numpy.ndarray` with
        integrated low dimensional embeddings and a single list of genes
        containing the intersection of inputted genes.
    """
    np.random.seed(seed)
    scanorama.random.seed(seed)

    datasets_full = scanorama.check_datasets(datasets_full)

    datasets, genes = scanorama.merge_datasets(datasets_full, genes_list,
                                               ds_names=ds_names, union=union)
    datasets_dimred, genes = scanorama.process_data(datasets, genes, hvg=hvg,
                                                    dimred=dimred)

    if sketch:
        datasets_dimred = scanorama.integrate_sketch(
            datasets_dimred, sketch_method=sketch_method, N=sketch_max,
            integration_fn=assemble, integration_fn_args={
                'verbose': verbose, 'knn': knn, 'sigma': sigma,
                'approx': approx, 'alpha': alpha, 'ds_names': ds_names,
                'batch_size': batch_size,
            },
        )

    else:
        datasets_dimred = assemble(
            datasets_dimred,  # Assemble in low dimensional space.
            verbose=verbose, knn=knn, sigma=sigma, approx=approx,
            alpha=alpha, ds_names=ds_names, batch_size=batch_size,
            alignments=alignments,  # EXTRA ARGUMENT I ADDED.
        )

    return datasets_dimred, genes


# Finds alignments between datasets and uses them to construct
# panoramas. "Merges" datasets by correcting gene expression
# values.
def assemble(datasets, verbose=VERBOSE, view_match=False, knn=KNN,
             sigma=SIGMA, approx=APPROX, alpha=ALPHA, expr_datasets=None,
             ds_names=None, batch_size=None,
             alignments=None, matches=None):
    if len(datasets) == 1:
        return datasets

    #### Edited here, the original code:
    # if alignments is None and matches is None:
    #    alignments, matches = find_alignments(
    #        datasets, knn=knn, approx=approx, alpha=alpha, verbose=verbose,
    #    )
    #### My edits...
    print(f"Using alignments: \n{alignments}\n")
    if matches is None:
        _, matches = scanorama.find_alignments(
            datasets, knn=knn, approx=approx, alpha=alpha, verbose=verbose,
        )

    ds_assembled = {}
    panoramas = []
    for i, j in alignments:
        if verbose:
            if ds_names is None:
                print('Processing datasets {}'.format((i, j)))
            else:
                print('Processing datasets {} <=> {}'.
                      format(ds_names[i], ds_names[j]))

        # Only consider a dataset a fixed amount of times.
        if not i in ds_assembled:
            ds_assembled[i] = 0
        ds_assembled[i] += 1
        if not j in ds_assembled:
            ds_assembled[j] = 0
        ds_assembled[j] += 1
        if ds_assembled[i] > 3 and ds_assembled[j] > 3:
            continue

        # See if datasets are involved in any current panoramas.
        panoramas_i = [panoramas[p] for p in range(len(panoramas))
                       if i in panoramas[p]]
        assert (len(panoramas_i) <= 1)
        panoramas_j = [panoramas[p] for p in range(len(panoramas))
                       if j in panoramas[p]]
        assert (len(panoramas_j) <= 1)

        if len(panoramas_i) == 0 and len(panoramas_j) == 0:
            if datasets[i].shape[0] < datasets[j].shape[0]:
                i, j = j, i
            panoramas.append([i])
            panoramas_i = [panoramas[-1]]

        # Map dataset i to panorama j.
        if len(panoramas_i) == 0:
            curr_ds = datasets[i]
            curr_ref = np.concatenate([datasets[p] for p in panoramas_j[0]])

            match = []
            base = 0
            for p in panoramas_j[0]:
                if i < p and (i, p) in matches:
                    match.extend([(a, b + base) for a, b in matches[(i, p)]])
                elif i > p and (p, i) in matches:
                    match.extend([(b, a + base) for a, b in matches[(p, i)]])
                base += datasets[p].shape[0]

            ds_ind = [a for a, _ in match]
            ref_ind = [b for _, b in match]

            bias = scanorama.transform(curr_ds, curr_ref, ds_ind, ref_ind,
                                       sigma=sigma,
                                       batch_size=batch_size)
            datasets[i] = curr_ds + bias

            if expr_datasets:
                curr_ds = expr_datasets[i]
                curr_ref = vstack([expr_datasets[p]
                                   for p in panoramas_j[0]])
                bias = scanorama.transform(curr_ds, curr_ref, ds_ind, ref_ind,
                                           sigma=sigma, cn=True,
                                           batch_size=batch_size)
                expr_datasets[i] = curr_ds + bias

            panoramas_j[0].append(i)

        # Map dataset j to panorama i.
        elif len(panoramas_j) == 0:
            curr_ds = datasets[j]
            curr_ref = np.concatenate([datasets[p] for p in panoramas_i[0]])

            match = []
            base = 0
            for p in panoramas_i[0]:
                if j < p and (j, p) in matches:
                    match.extend([(a, b + base) for a, b in matches[(j, p)]])
                elif j > p and (p, j) in matches:
                    match.extend([(b, a + base) for a, b in matches[(p, j)]])
                base += datasets[p].shape[0]

            ds_ind = [a for a, _ in match]
            ref_ind = [b for _, b in match]

            bias = scanorama.transform(curr_ds, curr_ref, ds_ind, ref_ind,
                                       sigma=sigma,
                                       batch_size=batch_size)
            datasets[j] = curr_ds + bias

            if expr_datasets:
                curr_ds = expr_datasets[j]
                curr_ref = vstack([expr_datasets[p]
                                   for p in panoramas_i[0]])
                bias = scanorama.transform(curr_ds, curr_ref, ds_ind, ref_ind,
                                           sigma=sigma,
                                           cn=True, batch_size=batch_size)
                expr_datasets[j] = curr_ds + bias

            panoramas_i[0].append(j)

        # Merge two panoramas together.
        else:
            curr_ds = np.concatenate([datasets[p] for p in panoramas_i[0]])
            curr_ref = np.concatenate([datasets[p] for p in panoramas_j[0]])

            # Find base indices into each panorama.
            base_i = 0
            for p in panoramas_i[0]:
                if p == i: break
                base_i += datasets[p].shape[0]
            base_j = 0
            for p in panoramas_j[0]:
                if p == j: break
                base_j += datasets[p].shape[0]

            # Find matching indices.
            match = []
            base = 0
            for p in panoramas_i[0]:
                if p == i and j < p and (j, p) in matches:
                    match.extend([(b + base, a + base_j)
                                  for a, b in matches[(j, p)]])
                elif p == i and j > p and (p, j) in matches:
                    match.extend([(a + base, b + base_j)
                                  for a, b in matches[(p, j)]])
                base += datasets[p].shape[0]
            base = 0
            for p in panoramas_j[0]:
                if p == j and i < p and (i, p) in matches:
                    match.extend([(a + base_i, b + base)
                                  for a, b in matches[(i, p)]])
                elif p == j and i > p and (p, i) in matches:
                    match.extend([(b + base_i, a + base)
                                  for a, b in matches[(p, i)]])
                base += datasets[p].shape[0]

            ds_ind = [a for a, _ in match]
            ref_ind = [b for _, b in match]

            # Apply transformation to entire panorama.
            bias = scanorama.transform(curr_ds, curr_ref, ds_ind, ref_ind,
                                       sigma=sigma,
                                       batch_size=batch_size)
            curr_ds += bias
            base = 0
            for p in panoramas_i[0]:
                n_cells = datasets[p].shape[0]
                datasets[p] = curr_ds[base:(base + n_cells), :]
                base += n_cells

            if not expr_datasets is None:
                curr_ds = vstack([expr_datasets[p]
                                  for p in panoramas_i[0]])
                curr_ref = vstack([expr_datasets[p]
                                   for p in panoramas_j[0]])
                bias = scanorama.transform(curr_ds, curr_ref, ds_ind, ref_ind,
                                           sigma=sigma, cn=True,
                                           batch_size=batch_size)
                curr_ds += bias
                base = 0
                for p in panoramas_i[0]:
                    n_cells = expr_datasets[p].shape[0]
                    expr_datasets[p] = curr_ds[base:(base + n_cells), :]
                    base += n_cells

            # Merge panoramas i and j and delete one.
            if panoramas_i[0] != panoramas_j[0]:
                panoramas_i[0] += panoramas_j[0]
                panoramas.remove(panoramas_j[0])

        # Visualize.
        # if view_match:
        #    scanorama.plot_mapping(curr_ds, curr_ref, ds_ind, ref_ind)

    return datasets

################################################################################
                # For adding the results to the AnnData #
################################################################################
def add_scanorama_to_data(data, datas, datasets, suffix, resolution=1):
    """Adds scanorama dimensionality reduced space to data.."""
    ## Adding merged space back into the original data ##
    print("Merging corrected anndatas...", file=sys.stdout, flush=True)
    correct_data = datas[0].concatenate(datas[1:],
                                        batch_key='dataset',
                                        batch_categories=datasets)

    orig_obs_names = ['-'.join(bc.split('-')[::-1][1:][::-1]) for bc in
                      correct_data.obs_names]

    orig_obs_names = ['-'.join(bc.split('-')[::-1][1:][::-1]) for bc in
                      correct_data.obs_names]
    corrected_space = pd.DataFrame(correct_data.obsm['X_scanorama'],
                                   index=orig_obs_names)
    corrected_space = corrected_space.loc[data.obs_names, :].values

    data.obsm[f'X_{suffix}'] = corrected_space

    sc.pp.neighbors(data, use_rep=f'X_{suffix}')
    sc.tl.umap(data)

    # Cacheing the umap
    data.obsm[f'X_umap_{suffix}'] = data.obsm['X_umap']

    # Clustering
    sc.tl.leiden(data, resolution=resolution)
    print(len(np.unique(data.obs['leiden'].values)))

    data.obs[f'leiden_{suffix}'] = data.obs['leiden']
