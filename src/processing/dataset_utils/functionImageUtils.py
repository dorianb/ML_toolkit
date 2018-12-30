from PIL import Image
from skimage.filters import threshold_local
from matplotlib import pyplot as plt
import numpy as np


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


def load_image(image_path, grayscale=True, binarize=False, normalize=False, resize_dim=None):
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

    # Open image
    image = Image.open(image_path)

    # Resize image
    if isinstance(resize_dim, tuple):
        image = image.resize(resize_dim, resample=Image.BICUBIC)

    # Convert to grayscale and optionally binarize the image
    if grayscale:
        image = image.convert('L')
        img = np.array(image)

        if binarize:
            block_size = 35
            local_thresh = threshold_local(img, block_size, offset=10)
            img = img > local_thresh

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