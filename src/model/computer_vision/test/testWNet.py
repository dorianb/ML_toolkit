import unittest
import os

from computer_vision.classWNet import WNet


ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = "/mnt/3A5620DE56209C9F/Dorian/Formation/Kaggle/HumpBack whales/all/train"
METADATA_PATH = os.path.join(ROOT_PATH, "metadata")


class WNetTestCase(unittest.TestCase):

    def test_init(self):

        wnet_1 = WNet(
            batch_size=2, n_channel=3, initializer_name="zeros",
            padding="SAME", k=2, from_pretrained=False, optimizer_name="rmsprop",
            learning_rate=0.001, n_epochs=10, checkpoint_step=10,
            metadata_path=METADATA_PATH, logger=None, name="WNet", debug=False)
        self.assertIsNotNone(wnet_1)

    def test_load_example(self):

        wnet_1 = WNet(
            batch_size=2, n_channel=3, initializer_name="zeros",
            padding="SAME", k=2, from_pretrained=False, optimizer_name="rmsprop",
            learning_rate=0.001, n_epochs=10, checkpoint_step=10,
            grayscale=False, binarize=False, normalize=False, resize_dim=(224, 224),
            metadata_path=METADATA_PATH, logger=None, name="WNet", debug=False)

        example = os.path.join(DATASET_PATH, "0a0c1df99.jpg")
        image = wnet_1.load_example(example)
        self.assertEqual(image.shape, (224, 224, 1))

    def test_load_batch(self):

        wnet_1 = WNet(
            batch_size=2, n_channel=3, initializer_name="zeros",
            padding="SAME", k=2, from_pretrained=False, optimizer_name="rmsprop",
            learning_rate=0.001, n_epochs=10, checkpoint_step=10,
            grayscale=True, binarize=False, normalize=False, resize_dim=(224, 224),
            metadata_path=METADATA_PATH, logger=None, name="WNet", debug=False)

        examples = [
            os.path.join(DATASET_PATH, "0a0c1df99.jpg"),
            os.path.join(DATASET_PATH, "0a00c7a0f.jpg")
        ]

        images = wnet_1.load_batch(examples)

        self.assertEqual(images.shape, (2, 224, 224, 1))

    def test_fit(self):

        training_set = [
            os.path.join(DATASET_PATH, "0a0c1df99.jpg"),
            os.path.join(DATASET_PATH, "0a00c7a0f.jpg"),
            os.path.join(DATASET_PATH, "0a2b7202c.jpg"),
            os.path.join(DATASET_PATH, "0a40d7e53.jpg")
        ]

        validation_set = []

        wnet_1 = WNet(
            batch_size=1, n_channel=1, initializer_name="zeros",
            padding="SAME", k=2, from_pretrained=False, optimizer_name="rmsprop",
            learning_rate=0.001, n_epochs=10, checkpoint_step=10,
            grayscale=True, binarize=False, normalize=False, resize_dim=(224, 224),
            metadata_path=METADATA_PATH, logger=None, name="WNet", debug=True)

        wnet_1.fit(training_set, validation_set)
        self.assertTrue(True)

        wnet_2 = WNet(
            batch_size=1, n_channel=3, initializer_name="zeros",
            padding="SAME", k=2, from_pretrained=False, optimizer_name="rmsprop",
            learning_rate=0.001, n_epochs=1, checkpoint_step=10,
            grayscale=False, binarize=False, normalize=False, resize_dim=(224, 224),
            metadata_path=METADATA_PATH, logger=None, name="WNet", debug=True)

        wnet_2.fit(training_set, validation_set)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
