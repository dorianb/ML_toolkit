import pandas as pd
import numpy as np


def filter_correlated_variables(df, features_name, threshold=0.9):
    """
    filter variable from data set which overpass a correlation threshold.

    Args:
        df: a pandas dataframe data set
        features_name: a list of variable names
        threshold: the correlation threshold from which variables are filtered
    Returns:
        the features name filtered
    """
    features_to_keep = set(features_name)

    corr = df[features_name].corr(method="pearson").abs()
    for f1 in features_name:
        for f2 in features_name:
            if f1 != f2 and corr.loc[f1, f2] > threshold and f2 in features_to_keep and f1 in features_to_keep:
                print("%s is too much correlated to %s" % (f2, f1))
                features_to_keep.remove(f2)
    return list(features_to_keep)

def information_value(X, y):
    """
    Compute the information value of a categorical variable for a binary variable to predict.
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
        iv.append((dist_bad - dist_good) * np.log(dist_bad / dist_good))
    return sum(iv)


def fisher_score(X, y):
    """
    Compute the fisher score of a continuous variable for a binary variable to predict.

    Args:
        X: a pandas serie representing a continuous variable
        y: a pandas serie representing a binary variable to predict

    Returns:
        the fisher score of the variable
    """
    classes = y.unique()
    return abs(X[y == classes[1]].mean() - X[y == classes[0]].mean()) / \
           np.sqrt(X[y == classes[1]].var() + X[y == classes[0]].var())
