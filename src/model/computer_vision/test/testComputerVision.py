import unittest
import os
import numpy as np
from computer_vision.classComputerVision import ComputerVision

ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "src/model/computer_vision/data/256_ObjectCategories")


class ComputerVisionTestCase(unittest.TestCase):

    def test_load_image(self):
        computer_vision_1 = ComputerVision()
        image_path = os.path.join(DATASET_PATH, "002.american-flag", "002_0001.jpg")

        image = computer_vision_1.load_image(image_path, grayscale=True, binarize=False,
                                             normalize=False)
        self.assertEqual(image.shape, (328, 500, 1))

        image = computer_vision_1.load_image(image_path, grayscale=False, binarize=False,
                                             normalize=False)
        self.assertEqual(image.shape, (328, 500, 3))

        image = computer_vision_1.load_image(image_path, grayscale=True, binarize=True,
                                             normalize=False)
        self.assertEqual(image.shape, (328, 500, 1))
        self.assertListEqual(np.unique(image).tolist(), [0, 1])


if __name__ == '__main__':
    unittest.main()
