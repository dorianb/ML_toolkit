import unittest
import os
from dataset_utils.functionCaltech import get_image_label


class CaltechTestCase(unittest.TestCase):

    def test_get_image_label(self):
        root_path = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
        dataset_path = os.path.join(root_path, "src/model/computer_vision/data/256_ObjectCategories")
        result = get_image_label(dataset_path, debug=True)
        self.assertGreater(len(result), 30607)

if __name__ == '__main__':
    unittest.main()
