import unittest
import os
from computer_vision.classVgg import Vgg


ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "data", "256_ObjectCategories")
METADATA_PATH = os.path.join(ROOT_PATH, "metadata")

class VggTestCase(unittest.TestCase):

    def test_init(self):
        classes = {0: 'ambiguous', 1: 'bird', 2: 'sheep'}
        vgg_1 = Vgg(classes, batch_size=1, height=224, width=224, dim_out=10,
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=1,
                    is_encoder=True, validation_size=10,
                    metadata_path=METADATA_PATH, logger=None)
        self.assertIsNotNone(vgg_1)

    def test_load_example(self):
        classes = {0: 'ambiguous', 1: 'bird', 2: 'sheep'}
        vgg_1 = Vgg(classes, batch_size=1, height=224, width=224, dim_out=10,
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=1,
                    is_encoder=True, validation_size=10,
                    metadata_path=METADATA_PATH, logger=None)
        example = os.path.join(DATASET_PATH, "002.american-flag", "002_0001.jpg"), 2
        image, label = vgg_1.load_example(example)
        self.assertEqual(image.shape, (224, 224, 1))
        self.assertEqual(label.shape, (3,))
        self.assertEqual(label[2], 1)

    def test_load_batch(self):
        classes = {0: 'ambiguous', 1: 'bird', 2: 'sheep'}
        vgg_1 = Vgg(classes, batch_size=2, height=224, width=224, dim_out=10,
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=1,
                    metadata_path=METADATA_PATH,
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

    def test_fit(self):

        classes = {0: 'ambiguous', 1: 'bird', 2: 'sheep'}

        training_set = [
            (os.path.join(DATASET_PATH, "002.american-flag", "002_0001.jpg"), 2),
            (os.path.join(DATASET_PATH, "002.american-flag", "002_0002.jpg"), 2),
            (os.path.join(DATASET_PATH, "002.american-flag", "002_0001.jpg"), 2),
            (os.path.join(DATASET_PATH, "002.american-flag", "002_0002.jpg"), 2)
        ]
        validation_set = [
            (os.path.join(DATASET_PATH, "002.american-flag", "002_0001.jpg"), 2),
            (os.path.join(DATASET_PATH, "002.american-flag", "002_0002.jpg"), 2)
        ]

        vgg_1 = Vgg(classes, batch_size=2, height=224, width=224, dim_out=len(classes),
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=1,
                    checkpoint_step=1, metadata_path=METADATA_PATH,
                    is_encoder=False, validation_size=10, logger=None)
        vgg_1.fit(training_set, validation_set)
        self.assertTrue(True)

        vgg_2 = Vgg(classes, batch_size=2, height=224, width=224, dim_out=len(classes),
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=1,
                    metadata_path=METADATA_PATH, is_encoder=False,
                    validation_size=10, from_pretrained=True, logger=None)
        vgg_2.fit(training_set, validation_set)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
