import pandas as pd
import numpy as np


def information_value(X, y, power_prediction_threshold=0.02):
    """
    Compute the information value for a categorical variable and a binary variable to predict.

    Args:
        X: a pandas serie representing a categorical variable
        y: a pandas serie representing a binary variable to predict
        power_prediction_threshold: the power prediction measure from which X is considered
            explainable of y.
    Returns:
        a boolean meaning the variable is explainable or not in regard of the threshold
    """
    X = pd.Series(X)
    y = pd.Series(y)

    K = X.nunique()
    classes = y.unique()
    iv = [((y[X == k] == classes[1]).sum() - (y[X == k] == classes[0]).sum())
        * np.log((y[X == k] == classes[1]).sum() / (y[X == k] == classes[0]).sum())
          for k in K].sum()
    return iv > power_prediction_threshold


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
