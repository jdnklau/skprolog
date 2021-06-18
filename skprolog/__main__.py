from skprolog import translate_tree, translate_forest
from skprolog import classifiers

from sklearn import datasets

if __name__ == "__main__":
    tree = classifiers.get_random_forest(datasets.fetch_covtype())
    print(translate_forest(tree))
