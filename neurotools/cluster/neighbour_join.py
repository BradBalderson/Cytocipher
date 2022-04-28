"""
Progressively merge clusters using neighbour-joining algorithm, measuring
distance between clusters as the no. of DE genes between them!

Copy-pasted from TreeMethods:

            https://github.com/BradBalderson/TreeMethods
"""

import numpy as np
import pandas as pd
import scanpy as sc

import scipy.spatial as spatial
from sklearn.neighbors import DistanceMetric

import neurotools.cluster.over_cluster as chs
from .cluster_helpers import n_different

def get_de_dists(data, expr_sub, col_key='leiden', knn: int=5,
                 logfc_cutoff: float=2, padj_cutoff: float=.001,
                 label_subset: np.array=None):
    """ Gets distances based on differential expression.
    """
    labels = data.obs[col_key].values
    label_set = np.array(list(data.obs[col_key].cat.categories))
    avg_data = chs.average(expr_sub, labels, label_set)
    if type(label_subset)==type(None):
        label_subset = label_set

    dist_mat = np.full((len(label_set), len(label_set)),
                       fill_value=avg_data.shape[1])
    point_tree = spatial.cKDTree(avg_data)
    for label in label_subset:
        i = np.where(label_set==label)[0][0]
        ##### Getting k-nearest neighbour clusters
        nearest_indices = point_tree.query(avg_data[i, :], k=knn + 1)[1][1:]
        nearest_labels = [label_set[index] for index in nearest_indices]

        ##### Calling de genes
        sc.tl.rank_genes_groups(data, groupby=col_key, use_raw=False,
                                groups=nearest_labels, reference=label,
                                )

        #### Getting results
        logfcs_rank = pd.DataFrame(
            data.uns['rank_genes_groups']['logfoldchanges']
        ).values.astype(float)
        padjs_rank = pd.DataFrame(data.uns['rank_genes_groups']['pvals_adj']
                                  ).values.astype(float)

        # Since comparing this label is the reference, need to look at
        up_bool = np.logical_and(logfcs_rank < -logfc_cutoff,
                                 padjs_rank < padj_cutoff)
        n_des = up_bool.sum(axis=0)
        dist_mat[i, nearest_indices] = n_des

    return dist_mat

def constructScores(sims, avgSims):
    """Constructs the scoring matrix taking into account distance between \
    clusters and each cluster and every other cluster.
    Params:
        sims (np.ndarray<float>): A matrix storing the pairwise distances \
                              between objects.
        avgSims (np.ndarray<float>): A matrix storing the average distance \
                                between a given thing and all other things.
    Returns:
        np.ndarray<float>: A matrix with the same shape of sims giving the \
                            distance scores.
    """

    scores = np.zeros(sims.shape)
    nClusters = scores.shape[0]
    for i in range(nClusters):
        for j in range(i + 1, nClusters):
            score = (nClusters - 2) * sims[i, j] + (-avgSims[i] - avgSims[j])
            scores[i, j] = scores[j, i] = score

    return scores

def updateSimMatrix(data, newLabels):
    """Updates the similarity matrix by fusing the relevant nodes, \
       producing updated labels as well
    Params:
        sims (np.ndarray<float>): A matrix storing the pairwise distances \
                              between objects.
        avgSims (np.ndarray<float>): A matrix storing the average distance \
                                between a given thing and all other things.
        i (int): Row index which is selected for fusion.
        j (int): Column index which is selected for fusion.
        labels (np.array<str>): The labels for the rows and columns.
        nIter (int): Keeps track of the current iteration.
    Returns:
        np.ndarray<float>: Updated sim matrix with the sims.shape-1 size, \
                                  since two of the pairs were fused to one node.
    """

    # Making the new similarity matrix, with the new node representing the fusion
    # between i and j represented on the first row and column
    # newSims = np.zeros((sims.shape[0] - 1, sims.shape[0] - 1))
    #
    # newSims[0, :] = [0] + [(sims[i, k] + sims[j, k] - sims[i, j]) / 2 for k in
    #                        remaining]
    # newSims[:, 0] = newSims[0, :]
    # # Rest of the matrix is the same, so just repopulating:
    # for m in range(1, newSims.shape[0]):
    #     for n in range(m + 1, newSims.shape[1]):
    #         newSims[m, n] = sims[remaining[m - 1], remaining[n - 1]]
    #         newSims[n, m] = newSims[m, n]
    label_values = data.obs['nj_clusters'].values
    expr_sub = data.layers['scaled']
    avg_data = chs.average(expr_sub, label_values, newLabels)

    dist = DistanceMetric.get_metric('manhattan')
    newSims = dist.pairwise(avg_data)

    # recalculating the average distance to the other clusters
    newAvgSims = np.apply_along_axis(sum, 1, newSims)

    # recalculating the scores
    newScores = constructScores(newSims, newAvgSims)

    return newSims, newAvgSims, newScores

def nj(data, groupby, max_de, padj_cutoff, logfc_cutoff,
       sims, avgSims, scores, labels):
    """Performs one iteration of neighbour joining tree construction, recursive.
    Params:
        sims (np.ndarray<float>): A matrix storing the pairwise distances \
                                                                between objects.
        tree (str): The current newick formatted tree being built.
        avgSims (np.ndarray<float>): A matrix storing the average distance \
                                     between a given thing and all other things.
        scores (np.ndarray<float>): A matrix with the same shape of sims \
                                                     giving the distance scores.
        labels (np.array<str>): The labels for the rows and columns.
        nIter (int): Keeps track of the current iteration.
        dist (bool): Whether the sims stores similarities or distances.
    Returns:
        str: A newick formatted tree representing the neighbour joining tree.
    """

    # in the final iteration, just need to add the relevant distance to the final
    # branch
    if (scores.shape[0] == 3):
        ##The final 3 will consist of the rest of the tree, and two other nodes
        # Need to just calculate the lengths for each, then append those two
        # new nodes to the rest of the tree with appropriate lengths..
        restToNode1 = sims[0, 1]  # X
        restToNode2 = sims[0, 2]  # Y
        node1ToNode2 = sims[1, 2]  # Z

        restBranchLen = (restToNode1 + restToNode2) - node1ToNode2
        node1BranchLen = restToNode1 - restBranchLen
        node2BranchLen = restToNode2 - restBranchLen

        newTree = "(" + labels[1] + ":" + str(node1BranchLen) + ", " + labels[
            0] + ":" + \
                  str(restBranchLen) + ", " + labels[2] + ":" + str(
            node2BranchLen) + ");"

        return newTree

    # Getting where the max score occurs
    maxScore = np.min(scores)

    locs = np.where(scores == maxScore)
    i, j = locs[0][0], locs[1][0]

    # Calculating if the two groupings are significantly different from
    # one another.
    orig_labels = data.obs[groupby].values
    orig_label_set = np.unique( data.obs[groupby].values )
    node_cluster_labels = np.array(['node '] * data.shape[0])
    for nodei, node in enumerate([labels[i], labels[j]]):
        node_clusters = [label for label in orig_label_set
                         if f" {label}:" in node or f"({label}:" in node or \
                         label==node]
        for cluster in node_clusters:
            node_cluster_labels[orig_labels == cluster] = f'node{nodei}'
    data.obs['node_labels'] = node_cluster_labels
    data.obs['node_labels'] = data.obs['node_labels'].astype('category')

    pair = ['node0', 'node1']
    n_de = n_different(data, 'node_labels', pair, logfc_cutoff, padj_cutoff)
    sig_different = n_de > max_de

    # Getting the distances which represent the distance between the selected
    # nodes i and j and their ancestor
    if not sig_different:
        dists = [0, 0]
    else:
        di = (sims[i, j] / 2) + (avgSims[i] - avgSims[j]) / (
                2 * (sims.shape[0] - 2))
        dj = (sims[i, j] / 2) + (avgSims[j] - avgSims[i]) / (
                2 * (sims.shape[0] - 2))
        dists = [di, dj]

    # Getting the indices which remain
    remaining = [k for k in range(sims.shape[0]) if k != i and k != j]
    remainingLabels = labels[remaining]

    # Updating groupings
    newGroup = "(" + labels[j] + ":" + str(dists[1]) + ", " + labels[i] + ":" + \
                                                             str(dists[0]) + ")"
    newLabels = np.array([newGroup] + list(remainingLabels))

    # Updating the joined labels...
    nj_clusters = data.obs['nj_clusters'].values.astype('object')
    nj_clusters[node_cluster_labels != 'node '] = newGroup
    data.obs['nj_clusters'] = nj_clusters

    # calculating the dists
    newSims, newAvgSims, newScores = updateSimMatrix(data, newLabels)

    # running another recursion
    return nj(data, groupby, max_de, padj_cutoff, logfc_cutoff,
              newSims, newAvgSims, newScores, newLabels)

def neighbour_join(data, groupby, max_de, padj_cutoff, logfc_cutoff):
    """Generates a tree via the neighbour joining method.
    Params:
        sims (np.ndarray<float>): A matrix storing the pairwise distances \
                                                                between objects.
        labels (np.array<str>): The labels for the rows and columns.
        dist (bool): Whether the sims stores similarities or distances.
    Returns:
        str: A newick formatted tree representing the neighbour joining tree.
    """

    expr_sub = data.to_df().copy()
    sc.pp.scale(expr_sub.values)
    data.layers['scaled'] = expr_sub

    labels = data.obs[groupby].values
    label_set = np.array(list(data.obs[groupby].cat.categories))
    avg_data = chs.average(expr_sub, labels, label_set)

    # Add this in to represent the joined clusters #
    data.obs['nj_clusters'] = data.obs[groupby].values

    dist = DistanceMetric.get_metric('manhattan')
    sims = dist.pairwise(avg_data)

    # average distance from each cluster to every other cluster
    avgSims = np.apply_along_axis(sum, 1, sims)

    scores = constructScores(sims, avgSims)

    return nj(data, groupby, max_de, padj_cutoff, logfc_cutoff,
              sims, avgSims, scores, label_set)

