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

## Prolog structures

### Decision trees

The generated decision tree Prolog structure has the following form:

```prolog
decision_tree(N_features, N_classes, Nodes, Feature_importances)
```

* `N_features` is the number of features used.
* `N_classes` is the number of output classes.
* `Nodes` is the structure of the tree.
  Each node is encoded either as `split_node/4` or `leaf/2` (see below)
* `Feature_importances` is the importance ranking of each feature;
  adds up to 1, higher values correspond to higher importance.

Inner nodes of the tree are of the form

```prolog
split_node(Feature, Threshold, Left_child, Right_child)
```

* `Feature` gives the feature index of the input which is inspected.
* `Threshold` is the splitting criterion. If the value of the inspected
  feature is less or equal to the threshold, the left child node is visited
  next. Otherwise the right node is visited next.
* `Left_child` and `Right_child` are either `split_node/4` themselves
  or `leaf/2`.

A leaf has the form

```prolog
leaf(Values)
```

* `Values` is a list of lists of integers.
  Each sublist counts the amount of
  samples during the training that reached this node. For a multi class
  problem with 4 classes, this could like like

  ```prolog
  Values = [[14, 9, 88, 13]]
  ```

  Multiple inner lists are used for multi-label classification.
  Each inner list corresponds to one predicted label.

The final prediction is the index (class) with highest recorded value.

### Random Forests

The generated Prolog structure for translated random forests has the form

```prolog
random_forest(N_featurs, N_classes, N_trees, Trees, Feature_importances)
```

* `N_features` is the number of features used.
* `N_classes` is the number of output classes.
* `N_trees` is the amount of trained trees.
* `Trees` is a list of translated decision trees as produced by
  `skprolog.translate_tree`.
* `Feature_importances` is the importance ranking of each feature;
  adds up to 1, higher values correspond to higher importance.

### Output example

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
