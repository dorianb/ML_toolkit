import unittest
import os
from computer_vision.classResNet import ResNet


ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "data", "Caltech256")
METADATA_PATH = os.path.join(ROOT_PATH, "metadata")

class ResNetTestCase(unittest.TestCase):

    def test_init(self):
        classes = {0: 'ambiguous', 1: 'bird', 2: 'sheep'}
        resnet_1 = ResNet(classes, n_layers=18, batch_size=1, height=224, width=224, dim_out=10,
                    grayscale=True, binarize=True, normalize=False,
                    learning_rate=10, n_epochs=1, validation_step=1,
                    is_encoder=False, validation_size=10,
                    metadata_path=METADATA_PATH, logger=None)
        self.assertIsNotNone(resnet_1)
        resnet_2 = ResNet(classes, n_layers=34, batch_size=1, height=224, width=224, dim_out=10,
                          grayscale=True, binarize=True, normalize=False,
                          learning_rate=10, n_epochs=1, validation_step=1,
                          is_encoder=False, validation_size=10,
                          metadata_path=METADATA_PATH, logger=None)
        self.assertIsNotNone(resnet_2)
        resnet_3 = ResNet(classes, n_layers=50, batch_size=1, height=224, width=224, dim_out=10,
                          grayscale=True, binarize=True, normalize=False,
                          learning_rate=10, n_epochs=1, validation_step=1,
                          is_encoder=False, validation_size=10,
                          metadata_path=METADATA_PATH, logger=None)
        self.assertIsNotNone(resnet_3)
        resnet_4 = ResNet(classes, n_layers=101, batch_size=1, height=224, width=224, dim_out=10,
                          grayscale=True, binarize=True, normalize=False,
                          learning_rate=10, n_epochs=1, validation_step=1,
                          is_encoder=False, validation_size=10,
                          metadata_path=METADATA_PATH, logger=None)
        self.assertIsNotNone(resnet_4)
        resnet_5 = ResNet(classes, n_layers=152, batch_size=1, height=224, width=224, dim_out=10,
                          grayscale=True, binarize=True, normalize=False,
                          learning_rate=10, n_epochs=1, validation_step=1,
                          is_encoder=False, validation_size=10,
                          metadata_path=METADATA_PATH, logger=None)
        self.assertIsNotNone(resnet_5)

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

        resnet_1 = ResNet(classes, n_layers=34, batch_size=1, height=56, width=56, dim_out=10,
                          grayscale=True, binarize=True, normalize=False,
                          learning_rate=10, n_epochs=1, validation_step=1,
                          is_encoder=False, validation_size=10, checkpoint_step=1,
                          metadata_path=METADATA_PATH, logger=None)
        resnet_1.fit(training_set, validation_set)
        self.assertTrue(True)

        resnet_2 = ResNet(classes, n_layers=34, batch_size=1, height=56, width=56, dim_out=10,
                          grayscale=True, binarize=True, normalize=False,
                          learning_rate=10, n_epochs=1, validation_step=1,
                          is_encoder=False, validation_size=10, from_pretrained=True,
                          metadata_path=METADATA_PATH, logger=None)
        resnet_2.fit(training_set, validation_set)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
