""" Functions for progressively merging clusters on tree if they are not
    significantly different from one another.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scanpy import AnnData

import ete3

from .tree_cluster_helpers import is_sig_different, is_sig_different_scores

def get_node_pairs(tree: ete3.Tree, initialise: bool=False):
    """Gets pairs of nodes which represent comparison between two distinct
        clusters (or merged clusters), & not comparisons between groups of
        clusters which have not been compared yet.
    """
    #### Initialise sig=False for every node
    if initialise:
        # Indicates whether cluster tested for significant difference
        for node in tree.get_descendants():
            node.sig = False

    ##### Finding each case where we have two clusters neighbouring one another.
    leaf_nodes = tree.get_leaves()
    selected_nodes = []
    node_pairs = []
    for node in leaf_nodes:
        if initialise:
            ##### Adding list of labels associated with each leaf node.
            ##### This important for merging clusters later...
            node.clusters = [node.name]
            node.name = 'False_' + node.name

        ##### Determining whether node is already paired with other cluster.
        ancestor = node.get_ancestors()[0]
        children = ancestor.get_children()
        # Must either be a leaf, or has to already be called as a significant
        # cluster if it's an internal node. That way atleast one is not significant,
        # so try to figure out if should merge it...
        # TODO figure out condition which will get a pair if
        #   is an internal node with two significant children..
        if np.all([#(child.is_leaf() or child.sig)
                   child.is_leaf()
                   and child not in selected_nodes
                   for child in children]):  # All children non-internal or found
            selected_nodes.extend(children)
            node_pairs.append(children)

    return node_pairs

def merge_iteration(data: AnnData, labels: np.array, node_pairs: list,
                    max_de: int = 2, padj_cutoff: float = .01,
                    logfc_cutoff: float = 2,
                    #### Just for testing the algorithm.
                    remove_sig_subtrees: bool=True,
                    ):
    """ Performs an iteration of merging pairs of clusters/already grouped
        clusters on the tree if they are not significantly different from
        one another!!
    """
    # max_label_size = max([len(label) for label in label_set])
    #new_node_pairs = []
    #i = 0
    sig_nodes = []
    for node_pair in node_pairs:
        node1, node2 = node_pair
        # print(f"iter {i}")
        # i += 1

        ### Labelling cells by node they are a part of
        node_cluster_labels = np.array(['node '] * data.shape[0])
        for cluster in node1.clusters:
            node_cluster_labels[labels == cluster] = 'node1'
        for cluster in node2.clusters:
            node_cluster_labels[labels == cluster] = 'node2'
        data.obs['node_labels'] = node_cluster_labels
        data.obs['node_labels'] = data.obs['node_labels'].astype('category')

        ### Determining if clusters significantly different
        sig_different, n_de = is_sig_different(data, 'node_labels',
                                                   'node1', 'node2',
                                                   max_de=max_de,
                                                   padj_cutoff=padj_cutoff,
                                                   logfc_cutoff=logfc_cutoff,
                                                   )

        ### If they aren't significantly different, then merge them!
        if not sig_different:
            try:
                ancestor = node1.get_ancestors()[0]
            except:
                print("Error! Returning node.")
                return node1, node2

            if hasattr(ancestor, 'clusters') and ancestor.sig:
                ##### For debugging during development only....
                ancestor.get_tree_root().render(
                               'plots/neurotools_dev/cluster/tree_at_error.png')
                raise Exception("Ancestor already merged...")

            ancestor.clusters = node1.clusters + node2.clusters
            ancestor.sig = False

            ### Add as new node pair...
            try:
                ancestor2 = ancestor.get_ancestors()[0]
            except:
                print("Error! Returning ancestor.")
                return ancestor
            ancestor2_children = ancestor2.get_children()
            other_child = \
            [node for node in ancestor2_children if node != ancestor][0]
            if not hasattr(other_child, 'clusters'):
                other_child_descendants = other_child.get_descendants()
                other_child_leaf_nodes = [node for node in
                                          other_child_descendants
                                          if node.is_leaf()]

                other_child.clusters = []
                for node in other_child_leaf_nodes:
                    other_child.clusters.extend( node.clusters )

                other_child.sig = False

            # Drop the child nodes, these are considered merged now.
            node1.detach()
            node2.detach()

            # Naming the new leaf nodes ... #
            for node in ancestor2_children:
                try:
                    node.name = 'False_'+'-'.join(node.clusters)
                except:
                    print("Error! Returning node.")
                    return node, node1, node2

            # Add new pairs to be evaluated, if they aren't already present!
            # if ancestor2_children not in new_node_pairs and \
            #                      ancestor2_children[::-1] not in new_node_pairs:
            #     for node in ancestor2_children:
            #         node.name = '-'.join(node.clusters)
            #
            #     new_node_pairs.append( ancestor2_children )

        ### If they sig different, label them as such!!!
        if sig_different:
            node1.sig = True
            node2.sig = True
            node1.name = node1.name.replace('False', 'True')
            node2.name = node2.name.replace('False', 'True')

            # ### Also set the ancestor node to true!!
            # ancestor = node1.get_ancestors()[0]
            # ancestor.sig = True
            # ancestor.name = ancestor.name.replace('False', 'True') \
            #              if 'False' in ancestor.name else 'True_'+ancestor.name

            """ 
            """
            if remove_sig_subtrees:
                ancestors = node1.get_ancestors()

                # Detaching this ancestor
                ancestor = ancestors[0]
                ancestor.detach()

                # Getting the ancestor one up, & it's children
                # (which no excludes our ancestor)
                ancestor2 = ancestors[1]
                ancestor2.detach()
                ancestor2_children = ancestor2.get_children()

                # Attaching the children to the ancestor one up #
                ancestor3 = ancestors[2]
                for child in ancestor2_children:
                    child.detach()
                    ancestor3.add_child(child)

            sig_nodes.extend( [node1, node2] )

    return sig_nodes

def check_different_iteration(data: AnnData, labels: np.array,
                              ancestor_node: ete3.TreeNode,
                    max_de: int = 2, padj_cutoff: float = .01,
                    logfc_cutoff: float = 2,):
    """ Top down approach to seeing if clusters significantly different..
    """
    ancestor_node.visited = True
    children = ancestor_node.get_children()
    if len(children) == 0: # Base case, at a root!!!
        tree = ancestor_node.get_tree_root()
        tree.cluster_groups.append( ancestor_node.clusters )
        return
    elif len(children) > 2:
        raise Exception("Greater than 2 children detected!!!! "
                        "Must be binary tree.")

    ### Getting the cluster labels for all clusters beneath nodes
    node_cluster_labels = np.array(['node '] * data.shape[0])
    for i, node in enumerate(children):
        node.clusters = [node.name for node in node.get_leaves()]
        for cluster in node.clusters:
            node_cluster_labels[labels == cluster] = f'node{i}'
    data.obs['node_labels'] = node_cluster_labels
    data.obs['node_labels'] = data.obs['node_labels'].astype('category')

    ### Determining if clusters significantly different
    sig_different, n_de = is_sig_different(data, 'node_labels',
                                           'node0', 'node1',
                                           max_de=max_de,
                                           padj_cutoff=padj_cutoff,
                                           logfc_cutoff=logfc_cutoff,
                                           )
    ancestor_node.sig = sig_different

    print(sig_different, n_de,
          len(children[0].clusters), len(children[1].clusters))

    if sig_different: # No merging, so keep going, preorder traversal!
        check_different_iteration(data, labels, children[0],
                                         max_de=max_de, padj_cutoff=padj_cutoff,
                                                    logfc_cutoff=logfc_cutoff)
        check_different_iteration(data, labels, children[1],
                                  max_de=max_de, padj_cutoff=padj_cutoff,
                                  logfc_cutoff=logfc_cutoff)
    else:
        tree = ancestor_node.get_tree_root()
        for node in children:
            tree.cluster_groups.append( node.clusters )







