import unittest
import os
from dataset_utils.functionCaltech import get_image_label, get_labels

ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "src/model/computer_vision/data/256_ObjectCategories")


class CaltechTestCase(unittest.TestCase):

    def test_get_image_label(self):
        result = get_image_label(DATASET_PATH)
        self.assertGreater(len(result), 30607)

    def test_get_labels(self):
        result = get_labels(DATASET_PATH)
        self.assertEqual(len(result), 257)
        self.assertEqual(result.keys()[0], 1)
        self.assertEqual(result.keys()[-1], 257)

if __name__ == '__main__':
    unittest.main()
