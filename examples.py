from skprolog import translate_tree, translate_forest

from sklearn import datasets

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def get_decision_tree(dataset):
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=123)

    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))

    return tree, train_acc, test_acc


def get_random_forest(dataset):
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=123)

    forest = RandomForestClassifier(n_estimators=50)
    forest.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, forest.predict(X_train))
    test_acc = accuracy_score(y_test, forest.predict(X_test))

    return forest, train_acc, test_acc


def write_example(file, data):
    tree, tree_train_acc, tree_test_acc  = get_decision_tree(data)
    tree = translate_tree(tree)
    forest, forest_train_acc, forest_test_acc = get_random_forest(data)
    forest = translate_forest(forest)

    with open(file, 'w+') as f:
        f.write("% skprolog example\n\n")
        f.write("%% Decision Tree\n\n")
        f.write(f"td_train_acc({tree_train_acc}).\n")
        f.write(f"td_test_acc({tree_test_acc}).\n")
        f.write(tree)
        f.write(".\n\n")
        f.write("%% Random Forest\n\n")
        f.write(f"rf_train_acc({forest_train_acc}).\n")
        f.write(f"rf_test_acc({forest_test_acc}).\n")
        f.write(forest)
        f.write(".\n")


if __name__ == "__main__":

    write_example("examples/iris.pl", datasets.load_iris())
    write_example("examples/breast_cancer.pl", datasets.load_breast_cancer())
    #write_example("examples/cover_type.pl", datasets.fetch_covtype()) # commented out as it takes very long
