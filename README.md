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

A basic usage example is given below.
[`usage_example.py`](usage_example.py) is a more detailed version of it,
especially useful if you are unfamiliar with the typical scikit-learn workflow
but want to extract the Prolog structures, e.g. for a Prolog-based analysis.

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

## Output example

A translated output from the Iris flower dataset might look like this:

```prolog
decision_tree(4, 3,
  split_node(2, 2.449999988079071,
    leaf([[37, 0, 0]]),
    split_node(3, 1.75,
      split_node(2, 5.349999904632568,
        split_node(0, 4.950000047683716,
          split_node(3, 1.350000023841858,
            leaf([[0, 1, 0]]),
            leaf([[0, 0, 1]])),
          split_node(2, 4.950000047683716,
            leaf([[0, 40, 0]]),
            split_node(1, 2.450000047683716,
              leaf([[0, 0, 1]]),
              leaf([[0, 2, 0]])))),
        leaf([[0,0,2]])),
      split_node(2, 4.8500001430511475,
        split_node(1, 3.0,
          leaf([[0, 0, 1]]),
          leaf([[0, 1, 0]])),
        leaf([[0, 0, 34]])))),
  [0.010888663256267358,
   0.02924587424274075,
   0.5451697757274787,
   0.41469568677351315])
```
