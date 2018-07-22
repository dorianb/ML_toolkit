import unittest
import os
from dataset_utils.CaltechDataset import CaltechDataset

ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "src/model/computer_vision/data/256_ObjectCategories")


class CaltechDatasetTestCase(unittest.TestCase):

    def test_init(self):
        caltech_dataset_1 = CaltechDataset(DATASET_PATH, train_size=0.7,
                                           val_size=0.2, test_size=0.1)

        self.assertLess(len(caltech_dataset_1.training_set), 30609 * 0.7)
        self.assertLess(len(caltech_dataset_1.training_set), 30609 * 0.2)
        self.assertLess(len(caltech_dataset_1.training_set), 30609 * 0.1)

    def test_get_examples(self):
        caltech_dataset_1 = CaltechDataset(DATASET_PATH)
        result = caltech_dataset_1.get_examples()
        self.assertGreater(len(result), 30607)

    def test_get_labels(self):
        caltech_dataset_1 = CaltechDataset(DATASET_PATH)
        result = caltech_dataset_1.get_labels()
        self.assertEqual(len(result), 257)
        self.assertEqual(result.keys()[0], 1)
        self.assertEqual(result.keys()[-1], 257)

if __name__ == '__main__':
    unittest.main()
