import numpy as np


def translate_tree(tree, node=0):
    """
    Translates a trained sklearn.tree.DecisionTreeClassifier
    into a Prolog structure.

    The returned structure has the following form:
        decision_tree(N_features, N_classes, Nodes)

    Here, `N_features` is the number of features used, `N_classes` is the
    number of output classes, and `Nodes` is the structure of the tree.
    Each node is encoded as follows:
        split_node(Feature, Threshold, Left_child, Right_child)

    `Feature` gives the feature index of the input which is inspected,
    `Threshold` is the splitting criterion. If the value of the inspected
    feature is less or equal to the threshold, the left child node is visited
    next. Otherwise the right node.
    `Left_child` and `Right_child` are either split nodes themselves or leaves.

    A leaf has the form
        leaf(Values)

    `Values` is a list of lists of integers. Each sublist counts the amount of
    samples during the training that reached this node. For a multi class
    problem with 4 classes, this could like like
        Values = [[14, 9, 88, 13]]

    The final prediction is the index (class) with highest recorded value.
    """

    treepl = translate_nodes(tree, node)

    if node == 0:
        treepl = f"decision_tree({tree.n_features_}, {tree.n_classes_}, {treepl})"

    return treepl

def translate_nodes(tree, node=0):
    tree_ = tree.tree_  # We want to access the internal structure for this

    # In the internal tree structure we have the following attributes:
    # * children_left[i]: id of the left child of node i or -1 if leaf node
    # * children_right[i]: id of the right child of node i or -1 if leaf node
    # * feature[i]: feature used for splitting node i
    # * threshold[i]: threshold value at node i
    # * n_node_samples[i]: the number of of training samples reaching node i
    # * impurity[i]: the impurity at node i
    #
    # Root node is node 0.

    if tree_.children_left[node] == -1:
        # Leaves are of the form `leaf(List)` where `List`
        # is a list of lists containing the number of training samples
        # per class which reached the leaf.
        # For multi-label instances `List` contains a list for each respective
        # class, according to Scikit-Learn documentation

        # Convert to int first for removing ".0" in string output
        leaf = np.array2string(tree_.value[node].astype(int),
                               separator=','
                               )
        return f"leaf({leaf})"
    else:
        # Split nodes are of the form `node(feature, threshold, left, right)`
        feature = tree_.feature[node]
        threshold = tree_.threshold[node]
        left = translate_tree(tree, node=tree_.children_left[node])
        right = translate_tree(tree, node=tree_.children_right[node])
        return f"split_node({str(int(feature))}, {threshold}, {left}, {right})"


def translate_forest(forest):
    """
    Translates a trained sklearn.ensemble.RandomForestClassifier
    into a Prolog structure.

    The returned structure has the following form:
        random_forest(N_featurs, N_classes, N_trees, Trees)

    Here, `N_features` is the number of features used, `N_classes` is the
    number of output classes, and `N_trees` is the amount of trained trees.
    `Trees` is a list of translated decision trees as produced by
    skprolog.translate_tree.
    """
    estimators = forest.estimators_
    trees = [translate_tree(t) for t in estimators]
    trees = ", ".join(trees)

    forestpl = f"random_forest({forest.n_features_}, {forest.n_classes_}, {len(estimators)}, [{trees}])"

    return forestpl
