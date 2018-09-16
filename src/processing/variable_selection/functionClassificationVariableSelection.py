import pandas as pd
import numpy as np


def information_value(X, y):
    """
    Compute the information value for a categorical variable and a binary variable to predict.
    Information value of a categorical variable has the following power prediction meaning:

        < 0.02 useless for prediction

        0.02 to 0.1 week predictor

        0.1 to 0.3 medium predictor

        0.3 to 0.5 strong predictor

        > 0.5 suspicious or to good to be true

    Args:
        X: a pandas serie representing a categorical variable
        y: a pandas serie representing a binary variable to predict

    Returns:
        a boolean meaning the variable is explainable or not in regard of the threshold
    """
    X = pd.Series(X)
    y = pd.Series(y)

    K = X.unique()
    classes = y.unique()

    iv = []
    for k in K:
        dist_good = (y[X == k] == classes[1]).sum() / (y == classes[1]).sum()
        dist_bad = (y[X == k] == classes[0]).sum() / (y == classes[0]).sum()
        iv.append((dist_good - dist_bad) / np.log(dist_good / dist_bad))
    return sum(iv)


def fisher_score(X, y):
    """
    Compute the fisher score of a continuous variable.

    Args:
        X: a pandas serie representing a continuous variable
        y: a pandas serie representing a binary variable to predict

    Returns:
        the fisher score of the variable
    """
    classes = y.unique()
    return abs(X[y == classes[1]].mean() - X[y == classes[0]].mean()) / \
           np.sqrt(X[y == classes[1]].var() + X[y == classes[0]].var())
