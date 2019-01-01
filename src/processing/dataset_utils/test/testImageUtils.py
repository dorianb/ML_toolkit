import unittest
import os

from dataset_utils.functionImageUtils import load_image


ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = "/mnt/3A5620DE56209C9F/Dorian/Formation/Kaggle/HumpBack whales/all/train"


class ImageUtilsTestCase(unittest.TestCase):

    def test_load_image(self):

        gray_image_path = os.path.join(DATASET_PATH, "0a0c1df99.jpg")

        gray_image = load_image(gray_image_path, grayscale=True, rgb=False,
                                binarize=False, normalize=False, resize_dim=(224, 224))

        self.assertTupleEqual(gray_image.shape, (224, 224, 1))

        gray_image = load_image(gray_image_path, grayscale=False, rgb=True,
                                binarize=False, normalize=False, resize_dim=(224, 224))

        self.assertTupleEqual(gray_image.shape, (224, 224, 3))

        gray_image = load_image(gray_image_path, grayscale=False, rgb=False,
                                binarize=False, normalize=False, resize_dim=(224, 224))

        self.assertTupleEqual(gray_image.shape, (224, 224, 1))

        color_image_path = os.path.join(DATASET_PATH, "0a00c7a0f.jpg")

        color_image = load_image(color_image_path, grayscale=True, rgb=False,
                                binarize=False, normalize=False, resize_dim=(224, 224))

        self.assertTupleEqual(color_image.shape, (224, 224, 1))

        color_image = load_image(color_image_path, grayscale=False, rgb=True,
                                binarize=False, normalize=False, resize_dim=(224, 224))

        self.assertTupleEqual(color_image.shape, (224, 224, 3))

        color_image = load_image(color_image_path, grayscale=False, rgb=False,
                                binarize=False, normalize=False, resize_dim=(224, 224))

        self.assertTupleEqual(color_image.shape, (224, 224, 3))

if __name__ == '__main__':
    unittest.main()
