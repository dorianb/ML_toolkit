import unittest
import numpy as np
from dataset_utils.classDataset import Dataset

class DatasetTestCase(unittest.TestCase):

    def test_train_val_test(self):
        """
        Test the train_val_test function

        """
        examples = ["path_" + str(i) for i in np.arange(0, 100)]

        dataset_1 = Dataset()
        train_set, val_set, test_set = dataset_1.train_val_test(
            examples, train_size=0.7, val_size=0.2, test_size=0.1)

        self.assertEqual(len(train_set), 70)
        self.assertEqual(len(val_set), 20)
        self.assertEqual(len(test_set), 10)

if __name__ == '__main__':
    unittest.main()
