from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=123)

def get_decision_tree():
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    return tree

if __name__ == "__main__":
    tree = get_decision_tree()

    acc_train = accuracy_score(y_train, tree.predict(X_train))
    acc_test = accuracy_score(y_test, tree.predict(X_test))
    print(f"Train set accuracy: {acc_train}")
    print(f"Test set accuracy:  {acc_test}")
