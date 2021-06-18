# Scikit Learn Prolog Trees

A translation unit for transpiling the
decision trees and random forests of
[Scikit Learn](https://scikit-learn.org/stable/)
into Prolog structures.

## Project dependencies

```bash
python -m venv env  # Create virtual env
source env/bin/activate  # activate virtual env
pip install -r requirements.txt
```

## Usage

```python
import skprolog
from sklearn.tree import DecisionTreeClassifier

# Prepare data
X_train, y_train = ...

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# Use skprolog
prolog_tree = skprolog.translate_tree(tree)

# prolog_tree is a string of the resulting Prolog structure
print(prolog_tree)
```

For `sklearn.ensemble.RandomForestClassifier` you can use
`skprolog.translate_forest(forest)`

## Examples

By running

```bash
python -m examples
```

you create example files in the `examples/` directory.
These train and translate a decision tree and a random forest each
on the following datasets:

* [Iris Flower dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
* [Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
* [Cover Type dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_covtype.html)
