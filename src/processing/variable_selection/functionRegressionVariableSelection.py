import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t


def explicability(weights, columns, first=10):
    """
    Representer le poids des variables explicatives
    """
    weights = weights.reshape(-1)
    idx = weights.argsort()[::-1][:first]
    x = np.arange(first)
    y = weights[idx]
    labels = columns[idx]

    plt.close('all')
    fig, ax = plt.subplots(figsize=(18, 4))

    ax.bar(x, y, width=0.5, align='center')
    ax.set_xlabel(u"Variables explicatives")
    ax.set_ylabel(u"Poids")

    plt.xticks(x, labels, rotation=80)
    plt.title(u"Poids des variables explicatives")
    plt.show()

    filename = "features_weights.csv"
    features_weights = []
    with open(filename, 'w') as f:
        for x in zip(labels, y):
            f.write(x[0])
            f.write("\t")
            f.write(str(x[1]))
            f.write("\n")
            features_weights.append(x)
    return labels


def score_with_components(clf, X, y, X_test, y_test, columns, weights, M=10, first=10):
    """
    Calculer les scores en fonction du nombre de composants
    """
    dict_feat_import = {
        'feature_names': list(columns),
        'feat_importance': list(weights)
    }

    feat_import_df = pd.DataFrame.from_dict(dict_feat_import, orient='columns')
    sort_feat_import_df = feat_import_df.sort_values(by=['feat_importance'], ascending=[0]).reset_index(drop=True)
    sorted_vect_importance = np.sort(weights, kind='heapsort')

    sorted_vect_importance[:] = sorted_vect_importance[::-1]
    score_mean = []
    error_on_mean_score = []
    n_th_most_important_features = []
    last_included_feat = []

    df_X = pd.DataFrame(X, columns=columns)
    df_y = pd.DataFrame(y)

    df_X_test = pd.DataFrame(X_test, columns=columns)
    df_y_test = pd.DataFrame(y_test)

    for ind in range(first):
        if ind % 10 == 0:
            print('ind = ' + ind.__str__())

        last_included_feat.append(list(sort_feat_import_df['feature_names'].head(ind + 1))[-1])
        X_new = df_X[list(sort_feat_import_df['feature_names'].head(ind + 1))]
        X_new_test = df_X_test[list(sort_feat_import_df['feature_names'].head(ind + 1))]

        scores = []
        for m in range(M):
            clf.fit(X_new, np.ravel(df_y))

            scores.append(clf.score(X_new_test, np.ravel(df_y_test)))

        error_on_mean_score.append(np.std(scores) / np.sqrt(M - 1))
        score_mean.append(np.mean(scores))
        n_th_most_important_features.append(ind + 1)

    return n_th_most_important_features, score_mean, error_on_mean_score, sort_feat_import_df


def plot_score_with_components(n_th_most_important_features, score_mean,
                               error_on_mean_score, sort_feat_import_df):
    """
    Tracer les scores en fonction du nombre de composants
    """
    x = n_th_most_important_features
    y = score_mean
    yerr = error_on_mean_score
    plt.clf()
    plt.close()
    plt.close('all')

    fig = plt.figure(figsize=(22.0, 13.0))
    plt.errorbar(x, y, yerr=yerr, fmt='o', linestyle='-', color='b')
    plt.title('mean score for n-th most important features', fontsize=22)
    ax = plt.gca()
    ax.legend_ = None
    plt.xlabel('nb of components considered ', fontsize=22)
    plt.ylabel('scores', fontsize=22)
    plt.legend(numpoints=1, loc=2)  # numpoints = 1 for nicer display
    axes = plt.gca()
    axes.set_xlim([0, sort_feat_import_df.shape[0]])
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.close()
    plt.close('all')


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