import unittest
import os
from computer_vision.classVgg import Vgg


ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "src/model/computer_vision/data/256_ObjectCategories")


class VggTestCase(unittest.TestCase):

    def test_init(self):
        classes = {0: 'ambiguous', 1: 'bird', 2: 'sheep'}
        vgg_1 = Vgg(classes, batch_size=1, height=1200, width=800, dim_out=10,
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=10,
                    is_encoder=True, validation_size=10, logger=None)
        self.assertIsNotNone(vgg_1)

    def test_load_example(self):
        classes = {0: 'ambiguous', 1: 'bird', 2: 'sheep'}
        vgg_1 = Vgg(classes, batch_size=1, height=1200, width=800, dim_out=10,
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=10,
                    is_encoder=True, validation_size=10, logger=None)
        example = os.path.join(DATASET_PATH, "002.american-flag", "002_0001.jpg"), 2
        image, label = vgg_1.load_example(example)
        self.assertEqual(image.shape, (224, 224, 1))
        self.assertEqual(label.shape, (3,))
        self.assertEqual(label[2], 1)

    def test_load_batch(self):
        classes = {0: 'ambiguous', 1: 'bird', 2: 'sheep'}
        vgg_1 = Vgg(classes, batch_size=2, height=1200, width=800, dim_out=10,
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=10,
                    is_encoder=True, validation_size=10, logger=None)
        examples = [
            (os.path.join(DATASET_PATH, "002.american-flag", "002_0001.jpg"), 2),
            (os.path.join(DATASET_PATH, "002.american-flag", "002_0002.jpg"), 2)
        ]
        images, labels = vgg_1.load_batch(examples)
        self.assertEqual(images.shape, (2, 224, 224, 1))
        self.assertEqual(labels.shape, (2, 3))
        self.assertEqual(labels[0][2], 1)
        self.assertEqual(labels[1][2], 1)


if __name__ == '__main__':
    unittest.main()
