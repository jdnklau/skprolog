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

    print(f"train accuracy: {accuracy_score(y_train, tree.predict(X_train))}")
    print(f"test accuracy: {accuracy_score(y_test, tree.predict(X_test))}")

    return tree


def get_random_forest(dataset):
    X = dataset.data
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        random_state=123)

    forest = RandomForestClassifier(n_estimators=50)
    forest.fit(X_train, y_train)

    print(f"train accuracy: {accuracy_score(y_train, forest.predict(X_train))}")
    print(f"test accuracy: {accuracy_score(y_test, forest.predict(X_test))}")

    return forest
