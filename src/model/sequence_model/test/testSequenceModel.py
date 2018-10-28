import unittest

from sequence_model.classSequenceModel import SequenceModel


class SequenceModelTestCase(unittest.TestCase):

    def test_init(self):
        """
        Test __init__ method
        """
        sm_1 = SequenceModel()

        self.assertTrue(sm_1.__class__ is SequenceModel)

    def test_get_optimizer(self):
        """
        Test get_optimizer method
        """
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