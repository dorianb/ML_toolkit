from PIL import Image
import tensorflow as tf
import numpy as np
from skimage.filters import threshold_local


class ComputerVision:

    def __init__(self):
        pass

    def build_model(self):
        pass

    def fit(self, training_set, validation_set):
        pass

    def load_batch(self, examples):
        pass

    def load_example(self, example):
        pass

    @staticmethod
    def load_image(image_path, grayscale=True, binarize=False, normalize=False,
                   resize_dim=None):
        """
        Load an image as an array

        Args:
            image_path: an image path
            grayscale: whether to load image as grayscale
            binarize: whether to binarize the image
            normalize: whether to normalize the image
            resize_dim: tuple (width, height) to which resize the image

        Returns:
            an array
        """

        """
        image_reader = tf.WholeFileReader()

        _, contents = image_reader.read(image_queue)

        image = tf.image.decode_jpeg(contents, channels=0)

        if isinstance(resize_dim, tuple):
            image = tf.image.resize_bicubic(image, resize_dim)

        if grayscale:
            image = tf.image.rgb_to_grayscale(image)

        if normalize:
            mean, var = tf.nn.moments(image, axes=(0, 1))
            image = tf.divide(
                tf.subtract(image, mean),
                tf.sqrt(var)
            )

        """
        image = Image.open(image_path)

        if isinstance(resize_dim, tuple):
            image = image.resize(resize_dim, resample=Image.BICUBIC)

        if grayscale:
            image = image.convert('L')
            img = np.array(image)

            if binarize:
                block_size = 35
                local_thresh = threshold_local(img, block_size, offset=10)
                img = img > local_thresh

            img = img.reshape((image.size[1], image.size[0], 1))

        else:
            img = np.array(image)

        if normalize:
            img = (img - np.mean(img, axis=(0, 1))) / np.std(img, axis=(0, 1))

        return img

    def validation_eval(self):
        pass

    def predict(self, set):
        pass

