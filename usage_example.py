"""
This file displays a more detailed usage example, targeted to Prolog
programmers who want to employ decision trees or random forests for their
projects or conduct a Prolog-based analysis over them.

Notes:

* The skprolog requirements do not include pandas. You need to install it
  yourself: `python -m pip install pandas`
* The code is an example for decision trees. For random forests you need to
  import `from sklearn.ensemble import RandomForestClassifier` and use
  `skprolog.translate_forest`.
"""
import skprolog

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# config
csv_path = r"path/to/csv"
target_column = "target"  # Column name from CSV for target variable
target_file = "example.pl"  # Where to store the prolog structure?

# Prepare data
data = pd.read_csv(csv_path)
# data = pd.get_dummies(data) # comment in to obtain One Hot encodings
X = data.drop([target_column], axis=1)
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=123)


# Train and evaluate classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))

print(f"Training accuracy: {train_acc : .3f}")
print(f"Test accuracy:     {test_acc : .3f}")


# Store prolog structure
prolog_tree = skprolog.translate_tree(clf)

print("Translated prolog structure: ")
print(prolog_tree)

with open(target_file, 'w+') as pl:
    pl.write(prolog_tree)
    pl.write('.')  # Terminating dot
