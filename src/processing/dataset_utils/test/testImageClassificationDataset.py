import unittest
import os
from dataset_utils.classImageClassificationDataset import ImageClassificationDataset

ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "data", "Caltech256")


class ImageClassificationDatasetTestCase(unittest.TestCase):

    def test_init(self):
        caltech_dataset_1 = ImageClassificationDataset(
            DATASET_PATH, train_size=0.7, val_size=0.2, test_size=0.1)

        self.assertLess(len(caltech_dataset_1.training_set), 30609 * 0.7)
        self.assertLess(len(caltech_dataset_1.validation_set), 30609 * 0.2)
        self.assertLess(len(caltech_dataset_1.test_set), 30609 * 0.1)

    def test_get_training_examples(self):
        caltech_dataset_1 = ImageClassificationDataset(DATASET_PATH)
        result = caltech_dataset_1.get_training_examples(absolute_path=True)
        self.assertGreater(len(result), 30607)

    def test_get_labels(self):
        caltech_dataset_1 = ImageClassificationDataset(DATASET_PATH)
        result = caltech_dataset_1.get_labels()
        self.assertEqual(len(result), 257)
        self.assertEqual(result.keys()[0], 1)
        self.assertEqual(result.keys()[-1], 257)

if __name__ == '__main__':
    unittest.main()
