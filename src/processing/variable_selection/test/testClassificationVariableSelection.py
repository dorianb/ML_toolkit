import unittest
import pandas as pd
import numpy as np
from variable_selection.functionClassificationVariableSelection import information_value, fisher_score


class ClassificationVariableSelectionTestCase(unittest.TestCase):

    def test_information_value(self):
        """
        Test the information_value function

        """
        df = pd.DataFrame({'A': ['spam', 'eggs', 'spam', 'eggs'] * 6,
                           'y': ['alpha', 'gamma'] * 12})

        iv_A= information_value(X=df["A"], y=df["y"])

        self.assertIsInstance(iv_A, float)
        self.assertGreaterEqual(iv_A, 0)
        self.assertLessEqual(iv_A, 1)

    def test_fisher_score(self):
        """
        Test the fisher_score function

        """
        df = pd.DataFrame({'A': np.random.randn(24),
                           'B': np.random.random_integers(0, 12, 24),
                           'y': [0, 1] * 12})

        iv_A = fisher_score(X=df["A"], y=df["y"])
        iv_B = fisher_score(X=df["B"], y=df["y"])

        self.assertIsInstance(iv_A, float)
        self.assertIsInstance(iv_B, float)

if __name__ == '__main__':
    unittest.main()
