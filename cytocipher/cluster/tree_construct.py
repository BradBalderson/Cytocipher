"""
Methods for the tree construction used to progressively merge clusters !

Copy-pasted from TreeMethods:

            https://github.com/BradBalderson/TreeMethods
"""

import numpy as np

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


def updateSimMatrix(sims, avgSims, i, j, labels, nIter):
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

    # Getting the distances which represent the distance between the selected
    # nodes i and j and their ancestor
    di = (sims[i, j] / 2) + (avgSims[i] - avgSims[j]) / (
                2 * (sims.shape[0] - 2))
    dj = (sims[i, j] / 2) + (avgSims[j] - avgSims[i]) / (
                2 * (sims.shape[0] - 2))

    # Getting the indices which remain
    remaining = [k for k in range(sims.shape[0]) if k != i and k != j]

    # Making the new similarity matrix, with the new node representing the fusion
    # between i and j represented on the first row and column
    newSims = np.zeros((sims.shape[0] - 1, sims.shape[0] - 1))

    newSims[0, :] = [0] + [(sims[i, k] + sims[j, k] - sims[i, j]) / 2 for k in
                           remaining]
    newSims[:, 0] = newSims[0, :]
    # Rest of the matrix is the same, so just repopulating:
    for m in range(1, newSims.shape[0]):
        for n in range(m + 1, newSims.shape[1]):
            newSims[m, n] = sims[remaining[m - 1], remaining[n - 1]]
            newSims[n, m] = newSims[m, n]

    # Updating the labels so can label the tree appropriately
    remainingLabels = labels[remaining]

    # recalculating the average distance to the other clusters
    newAvgSims = np.apply_along_axis(sum, 1, newSims)

    # recalculating the scores
    newScores = constructScores(newSims, newAvgSims)

    return newSims, newAvgSims, newScores, remainingLabels, [di, dj]


def nj(sims, tree, avgSims, scores, labels, nIter, dist=False):
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
    if not dist:
        maxScore = np.max(scores)
    else:
        maxScore = np.min(scores)

    locs = np.where(scores == maxScore)
    i, j = locs[0][0], locs[1][0]

    # calculating the dists
    newSims, newAvgSims, newScores, remainingLabels, dists = \
        updateSimMatrix(sims, avgSims, i, j, labels, nIter)

    # Updating tree
    newTree = "(" + labels[j] + ":" + str(dists[1]) + ", " + labels[i] + ":" + \
              str(dists[0]) + ")"
    newLabels = np.array([newTree] + list(remainingLabels))

    # running another recursion
    return nj(newSims, newTree, newAvgSims, newScores, newLabels, nIter + 1,
              dist=dist)


def njTree(sims, labels, dist=True):
    """Generates a tree via the neighbour joining method.
    Params:
        sims (np.ndarray<float>): A matrix storing the pairwise distances \
                                                                between objects.
        labels (np.array<str>): The labels for the rows and columns.
        dist (bool): Whether the sims stores similarities or distances.
    Returns:
        str: A newick formatted tree representing the neighbour joining tree.
    """

    # Initialising a tree with all nodes connected
    tree = ''

    # average distance from each cluster to every other cluster
    avgSims = np.apply_along_axis(sum, 1, sims)

    scores = constructScores(sims, avgSims)

    return nj(sims, tree, avgSims, scores, labels, 0, dist=dist)

