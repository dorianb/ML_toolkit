import numpy as np
from scipy.stats import t


def compute_tstats(X, beta, u):
    """
    Compute student statistics.

    Args:
        X: a numpy array with n x p dimension (n representing observations and p features)
        beta: a numpy array representing regression coefficients.
        u: a numpy array representing residuals for regression.

    Returns:
        features student statistics
    """
    n = u.shape[0]
    p = beta.shape[0]

    sigma = np.dot(u.T, u) / (n - p)
    std = np.sqrt(sigma * np.linalg.inv(np.dot(X.T, X)).diagonal())

    return beta / std


def ols(X, y, with_intercept=True, standardize=True):
    """
    Estimate the regression coefficients using an ordinary least squared algorithm.

    Args:
        X: a numpy array with n x p dimension (n representing observations and p features)
        y: a numpy array with n x 1 dimension (n representing observations and the other dimension the target variable)
        with_intercept: whether to add an intercept
        standardize: whether to standardize features
    Returns:
        a triplet with regression coefficients, residuals and t stats
    """
    n, p = X.shape

    # Standard scaling
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) if standardize else X

    # Add intercept
    X = np.concatenate([np.ones(n).reshape(-1, 1), X],
                       axis=1) if with_intercept else X

    # Estimate the regression coefficients
    beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    # Predict y
    y_pred = np.dot(X, beta)

    # Compute residuals
    u = y_pred - y

    # Features significance
    t_stats = compute_tstats(X, beta, u)

    return beta, u, t_stats


def backward_elimination(X, y, alpha_critic=0.2):
    """
    Apply backward elemination step-wise method.

    Args:
        X: a numpy array with n x p dimension (n representing observations and p features)
        y: a numpy array with n x 1 dimension (n representing observations and the other dimension the target variable)
        alpha_critic: confidence threshold

    Returns:
        a list of boolean with p-dimension (number of features) saying whether the variable is of interest or not.

    """
    n, p = X.shape
    kept_features = np.ones(p).tolist()
    is_significant = np.zeros(p).tolist()

    while not all(is_significant):

        features = [f for f in range(p) if kept_features[f]]
        _, _, t_stats = ols(X[:, features], y, with_intercept=True, standardize=True)

        cv = t.ppf(1.0 - alpha_critic, n - (len(features) + 1))
        is_significant = (t_stats > cv)[1:]

        kept_features = [1 if f in features and is_significant[features.index(f)] else 0 for f in range(p)]

    return kept_features


def forward_selection(X, y, alpha_critic=0.2):
    """
    Apply forward selection step-wise method.

    Args:
        X: a numpy array with n x p dimension (n representing observations and p features)
        y: a numpy array with n x 1 dimension (n representing observations and the other dimension the target variable)
        alpha_critic: confidence threshold

    Returns:
        a list of boolean with p dimension (number of features) saying whether the variable is of interest or not.

    """
    n, p = X.shape
    kept_features = np.zeros(p).tolist()
    is_significant = np.ones(p).tolist()

    while any(is_significant):
        features_to_add = [f for f in range(p) if not kept_features[f]]
        features = [f for f in range(p) if kept_features[f]]

        is_significant = []
        t_values = []

        for feature_to_add in features_to_add:
            _, _, t_stats = ols(X[:, features + [feature_to_add]], y,
                                with_intercept=True, standardize=True)

            cv = t.ppf(1.0 - alpha_critic, n - (len(features + [feature_to_add]) + 1))
            is_significant.append(t_stats[-1] > cv)
            t_values.append(t_stats[-1])

        kept_features = [1
                         if f in features
                         or
                         (
                             f in features_to_add
                             and is_significant[features_to_add.index(f)]
                             and t_values[features_to_add.index(f)] == max(t_values)
                         )
                         else 0 for f in range(p)]

    return kept_features

def stepwise_regression():
    pass

def optimal_criterian():
    pass