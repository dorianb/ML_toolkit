import os
from datetime import datetime
from PIL import Image
import tensorflow as tf
import numpy as np
from skimage.filters import threshold_local


class ComputerVision:

    def __init__(self, summary_path="", checkpoint_path=""):
        """
        The initalization of a computer vision model.

        Args:
            summary_path: the path to the summaries
            checkpoint_path: the path to the checkpoints
        """
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
        self.saver = None
        self.name = "computer_vision"

    def build_model(self):
        pass

    def fit(self, training_set, validation_set):
        pass

    def load_batch(self, examples):
        pass

    def load_example(self, example):
        pass

    @staticmethod
    def get_optimizer(name="adam", learning_rate=0.1):
        """
        Get the optimizer object corresponding. If unknown optimizer, raise an exception.

        Args:
            name: name of the optimizer
            learning_rate: the learning rate
        Returns:
            a tensorflow optimizer object
        """
        if name == "adam":
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif name == "adadelta":
            return tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        else:
            raise Exception("The optimizer is unknown")

    def get_writer(self):
        """
        Get the training and validation summaries writers.

        Returns:
            Tensorflow FileWriter object
        """
        training_path = os.path.join(self.summary_path, "train", str(datetime.now()))
        validation_path = os.path.join(self.summary_path, "val", str(datetime.now()))
        return tf.summary.FileWriter(training_path), \
               tf.summary.FileWriter(validation_path)

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

    def load(self, session):
        """

        Args:
            session: the tensorflow session

        Returns:
            Nothing
        """
        filename = [self.name + "-" + filename.split(self.name)[1].split("-")[1].split(".")[0]
                    for filename in sorted(os.listdir(self.checkpoint_path))
                    if self.name in filename].pop()
        checkpoint_path = os.path.join(self.checkpoint_path, filename)
        self.saver.restore(session, checkpoint_path)

    def save(self, session, step):
        """

        Args:
            session: the tensorflow session
            step: the global step as a tensor

        Returns:
            the path to the saved model
        """
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        checkpoint_path = os.path.join(self.checkpoint_path, self.name)
        return self.saver.save(session, checkpoint_path, global_step=step)

    def validation_eval(self):
        pass

    def predict(self, set):
        pass

