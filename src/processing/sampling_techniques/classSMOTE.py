import numpy as np
import pandas as pd


class SMOTE:
    """
    SMOTE: Synthetic Minority Over-sampling Technique.
    The purpose and algorithm is described onto the following paper: https://arxiv.org/pdf/1106.1813.pdf
    """

    def __init__(self, k):
        """
        SMOTE initialization.

        Args:
            k: nearest neighbours examples to consider
        """
        self.k = k

    def fit_resample(self, X, target_name="target", features=[]):
        """
        Fit and resample dataset.

        Args:
            X: a dataframe with imbalanced classes
            target_name: string representing the column name of the target
            features: list of column names features

        Returns:
            a dataframe containing over-sampled dataset

        """
        counting = X[target_name].value_counts().sort_values().reset_index().values
        minority_class, n_minority_class = counting[0]
        majority_class, n_majority_class = counting[1]

        N = n_majority_class - n_minority_class

        minority_samples = X.loc[X[target_name] == minority_class, features].values
        synthetic_samples = np.empty(shape=(N, len(features)))

        # Compute k nearest neighbours
        nearest_neighbours = {}
        it1 = np.nditer(minority_samples, flags=['f_index'])
        while not it1.finished:
            nn = {}
            mask = np.ones(minority_samples.shape, dtype=bool)
            mask[it1.index, :] = 0

            it2 = np.nditer(minority_samples[mask], flags=['f_index'])
            while not it2.finished:
                nn[it2.index] = sum([(it1[f] - it2[f]) ** 2 for f in range(len(features))])
                it2.iternext()

            nearest_neighbours[it1.index] = list(zip(*sorted(nn.items(), key=lambda x: x[1], reverse=True)[:self.k])[0])
            it1.iternext()

        # Generate synthetic samples
        nn = np.random.randint(0, self.k, N)
        gap = np.random.rand(N, len(features))
        while N > 0:
            neighbour = nearest_neighbours[N-1][nn[N-1]]
            diff = minority_samples[neighbour, :] - minority_samples[N-1, :]
            synthetic_samples[N-1, :] = minority_samples[N-1, :] + gap[N-1, :] * diff

        return pd.concat([X, pd.DataFrame(synthetic_samples, columns=features)])