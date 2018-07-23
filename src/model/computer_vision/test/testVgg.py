import unittest
from computer_vision.classVgg import Vgg


class VggTestCase(unittest.TestCase):

    def test_init(self):
        vgg_1 = Vgg(batch_size=1, height=1200, width=800, dim_out=10,
                    grayscale=True, binarize=True, normalize=False,
                    n_classes=10, learning_rate=10, n_epochs=1, validation_step=10,
                    is_encoder=True, validation_size=10, logger=None)
        self.assertIsNotNone(vgg_1)

    def test_load_example(self):
        self.assertEqual(True, True)

    def test_load_batch(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
