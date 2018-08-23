import unittest
import os
import numpy as np
import tensorflow as tf

from computer_vision.classComputerVision import ComputerVision

ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "data", "256_ObjectCategories")


class ComputerVisionTestCase(unittest.TestCase):

    def test_get_optimizer(self):

        computer_vision_1 = ComputerVision()
        opt_1 = computer_vision_1.get_optimizer(name="adam", learning_rate=0.2)
        self.assertEqual(type(opt_1).__name__, "AdamOptimizer")

        opt_2 = computer_vision_1.get_optimizer(name="adadelta", learning_rate=0.3)
        self.assertEqual(type(opt_2).__name__, "AdadeltaOptimizer")

        with self.assertRaises(Exception) as context:
            _ = computer_vision_1.get_optimizer(name="unknown")

        self.assertTrue("unknown" in str(context.exception))

    def test_load_image(self):

        sess = tf.InteractiveSession()

        computer_vision_1 = ComputerVision()
        image_path = os.path.join(DATASET_PATH, "002.american-flag", "002_0001.jpg")

        image = computer_vision_1.load_image(image_path, grayscale=False, binarize=False,
                                             normalize=False, resize_dim=None)
        # image = image.eval()

        self.assertEqual(image.shape, (328, 500, 3))

        image = computer_vision_1.load_image(image_path, grayscale=True, binarize=False,
                                             normalize=False, resize_dim=None)
        self.assertEqual(image.shape, (328, 500, 1))

        image = computer_vision_1.load_image(image_path, grayscale=True, binarize=True,
                                             normalize=False, resize_dim=None)
        self.assertEqual(image.shape, (328, 500, 1))
        self.assertListEqual(np.unique(image).tolist(), [0, 1])

        image = computer_vision_1.load_image(image_path, grayscale=True, binarize=False,
                                             normalize=False, resize_dim=(224, 224))
        self.assertEqual(image.shape, (224, 224, 1))

        sess.close()

if __name__ == '__main__':
    unittest.main()
