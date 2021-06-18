from skprolog import translate_tree
from skprolog.classifiers import iris

if __name__ == "__main__":
    tree = iris.get_decision_tree()
    print(translate_tree(tree))
