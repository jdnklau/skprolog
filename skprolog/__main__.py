from skprolog import translate_tree
from skprolog import classifiers

from sklearn import datasets

if __name__ == "__main__":
    tree = classifiers.get_decision_tree(datasets.load_iris())
    print(translate_tree(tree))
