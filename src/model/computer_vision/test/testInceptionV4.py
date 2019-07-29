import unittest
import os
from computer_vision.classInceptionV4 import InceptionV4


ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "data", "Caltech256")
METADATA_PATH = os.path.join(ROOT_PATH, "metadata")

class InceptionV4TestCase(unittest.TestCase):

    def test_init(self):
        classes = {0: 'ambiguous', 1: 'bird', 2: 'sheep'}
        inception_1 = InceptionV4(classes, batch_size=1, height=224, width=224, dim_out=10,
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=1,
                    is_encoder=False, validation_size=10,
                    metadata_path=METADATA_PATH, logger=None)
        self.assertIsNotNone(inception_1)

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

        inception_1 = InceptionV4(classes, batch_size=1, height=299, width=299, dim_out=10,
                          grayscale=True, binarize=True, normalize=False,
                          learning_rate=10, n_epochs=1, validation_step=1,
                          is_encoder=False, validation_size=10, checkpoint_step=1,
                          metadata_path=METADATA_PATH, logger=None)
        inception_1.fit(training_set, validation_set)
        self.assertTrue(True)

        inception_2 = InceptionV4(classes, batch_size=1, height=299, width=299, dim_out=10,
                          grayscale=True, binarize=True, normalize=False,
                          learning_rate=10, n_epochs=1, validation_step=1,
                          is_encoder=False, validation_size=10, from_pretrained=True,
                          metadata_path=METADATA_PATH, logger=None)
        inception_2.fit(training_set, validation_set)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
