from PIL import Image
from skimage.filters import threshold_local
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


def plot_image(image, gray=False):
    """
    Plot image.

    Args:
        image: numpy array representing an image
        gray: whether to display image in grayscale
    Returns:
        matplotlib axe
    """
    image = np.squeeze(image)
    cmap = "gray" if gray else None
    return plt.imshow(image, cmap=cmap)


def read_image(image_queue, grayscale=True, rgb=False, binarize=False, normalize=False, resize_dim=None, n_multiple_dim=None):
    """
    Read image using tensorflow operations.

    Args:
        image_queue: an image queue object
        grayscale: whether to load image as grayscale
        rgb: whether to load image as rgb
        binarize: whether to binarize the image
        normalize: whether to normalize the image
        resize_dim: tuple (width, height) to which resize the image
        n_multiple_dim: image will be resized to the nearest multiple

    Returns:
        an array
    """
    image_reader = tf.WholeFileReader()

    _, record_string = image_reader.read(image_queue)

    image = tf.image.decode_image(record_string, channels=0)

    if isinstance(resize_dim, tuple):
        image = tf.image.resize_bicubic(image, resize_dim)

    if n_multiple_dim:
        resize_dim = (
            tf.cond(
                tf.mod(image.shape[0], tf.constant(n_multiple_dim)) != 0,
                lambda: tf.round(image.shape[0] / n_multiple_dim),
                lambda: image.shape[0]
            ),
            tf.cond(
                tf.mod(image.shape[1], tf.constant(n_multiple_dim)) != 0,
                lambda: tf.round(image.shape[1] / n_multiple_dim),
                lambda: image.shape[1]
            )
        )
        image = tf.cond(
            tf.mod(image.shape[0], tf.constant(n_multiple_dim)) != 0
            or tf.mod(image.shape[1], tf.constant(n_multiple_dim)) != 0,
            lambda: tf.image.resize_bicubic(image, resize_dim),
            lambda: image
        )

    if grayscale:
        image = tf.image.rgb_to_grayscale(image)

    if rgb:
        image = tf.image.grayscale_to_rgb(image)

    if normalize:
        mean, var = tf.nn.moments(image, axes=(0, 1))
        image = tf.divide(
            tf.subtract(image, mean),
            tf.sqrt(var)
        )

    return image


def load_image(image_path, grayscale=True, rgb=False, binarize=False, normalize=False, resize_dim=None, n_multiple_dim=None):
    """
    Load an image as an array

    Args:
        image_path: an image path
        grayscale: whether to load image as grayscale
        rgb: whether to load image as rgb
        binarize: whether to binarize the image
        normalize: whether to normalize the image
        resize_dim: tuple (width, height) to which resize the image
        n_multiple_dim: image will be resized to the nearest multiple

    Returns:
        an array
    """
    # Open image
    image = Image.open(image_path)

    # Resize image
    if isinstance(resize_dim, tuple):
        image = image.resize(resize_dim, resample=Image.BICUBIC)

    if n_multiple_dim is not None and (image.size[0] % n_multiple_dim != 0 or image.size[1] % n_multiple_dim != 0):
        resize_dim = (
            int(image.size[0] / n_multiple_dim) * n_multiple_dim if image.size[0] % n_multiple_dim != 0 else image.size[0],
            int(image.size[1] / n_multiple_dim) * n_multiple_dim if image.size[1] % n_multiple_dim != 0 else image.size[1]
        )
        image = image.resize(resize_dim, resample=Image.BICUBIC)

    # Convert to grayscale and optionally binarize the image
    if grayscale:
        image = image.convert('L')
        img = np.array(image)

        if binarize:
            block_size = 35
            local_thresh = threshold_local(img, block_size, offset=10)
            img = img > local_thresh

    elif rgb:
        image = image.convert('RGB')
        img = np.array(image)

    else:
        img = np.array(image)

    # Reshape image from (width, height, channel) to (height, width, channel)
    if len(img.shape) == 2:
        img = img.reshape((image.size[1], image.size[0], 1))

    elif len(img.shape) == 3:
        img = img.reshape((image.size[1], image.size[0], img.shape[2]))

    # Normalize image
    if normalize:
        img = (img - np.mean(img, axis=(0, 1))) / np.std(img, axis=(0, 1))

    return img