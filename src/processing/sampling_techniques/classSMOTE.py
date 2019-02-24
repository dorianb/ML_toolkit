

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
        Fit and resample datatset.

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

