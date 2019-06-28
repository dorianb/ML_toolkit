import unittest
import os

from sequence_model.classSequenceModel import SequenceModel

ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-2])
DATASET_PATH = os.path.join(ROOT_PATH, "data", "e08_features_all_study_v1")


class SequenceModelTestCase(unittest.TestCase):

    def test_get_optimizer(self):

        sm_1 = SequenceModel()
        opt_1 = sm_1.get_optimizer(name="adam", learning_rate=0.2)
        self.assertEqual(type(opt_1).__name__, "AdamOptimizer")

        opt_2 = sm_1.get_optimizer(name="adadelta", learning_rate=0.3)
        self.assertEqual(type(opt_2).__name__, "AdadeltaOptimizer")

        with self.assertRaises(Exception) as context:
            _ = sm_1.get_optimizer(name="unknown")

        self.assertTrue("unknown" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
