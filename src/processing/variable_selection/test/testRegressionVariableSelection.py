import unittest
import numpy as np
from sklearn.datasets import load_boston

from variable_selection.functionRegressionVariableSelection import ols, backward_elimination, forward_selection


class RegressionVariableSelectionTestCase(unittest.TestCase):

    def test_ols(self):
        """
        Test the ols function.

        """
        X, y = load_boston(return_X_y=True)
        n, p = X.shape

        beta, u, t_stats = ols(X, y, with_intercept=True, standardize=True)

        self.assertTupleEqual(beta.shape, (p + 1,))
        self.assertTupleEqual(u.shape, (n,))
        self.assertTupleEqual(t_stats.shape, (p + 1,))

    def test_backward_elimination(self):
        """
        Test the backward elimination function.

        """
        X, y = load_boston(return_X_y=True)
        n, p = X.shape

        features_is_significant = backward_elimination(X, y, alpha_critic=0.2)

        self.assertEqual(len(features_is_significant), p)

        features_to_keep = [f for f in range(p) if features_is_significant[f]]
        print(features_to_keep)
        _, u1, _ = ols(X, y, with_intercept=True, standardize=True)
        _, u2, _ = ols(X[:, features_to_keep], y, with_intercept=True, standardize=True)

        mse1 = np.mean(u1 ** 2)
        mse2 = np.mean(u2 ** 2)

        self.assertGreater(mse1, mse2)

    def test_forward_selection(self):
        """
        Test the forward selection function.

        """
        X, y = load_boston(return_X_y=True)
        n, p = X.shape

        features_is_significant = forward_selection(X, y, alpha_critic=0.2)

        self.assertEqual(len(features_is_significant), p)

        features_to_keep = [f for f in range(p) if features_is_significant[f]]
        print(features_to_keep)
        _, u1, _ = ols(X, y, with_intercept=True, standardize=True)
        _, u2, _ = ols(X[:, features_to_keep], y, with_intercept=True, standardize=True)

        mse1 = np.mean(u1 ** 2)
        mse2 = np.mean(u2 ** 2)

        self.assertGreater(mse1, mse2)

if __name__ == '__main__':
    unittest.main()
