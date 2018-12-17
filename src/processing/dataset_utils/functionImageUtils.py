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