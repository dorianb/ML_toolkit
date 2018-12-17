import os
from datetime import datetime
from PIL import Image
import tensorflow as tf
import numpy as np
from skimage.filters import threshold_local


class ComputerVision:

    def __init__(self, summary_path="", checkpoint_path=""):
        """
        The initialization of a computer vision model.

        Args:
            summary_path: the path to the summaries
            checkpoint_path: the path to the checkpoints
        """
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
        self.global_step = None
        self.saver = None
        self.logger = None
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
    def get_initializer(name):
        """
        Get the initializer.

        Args:
            name: the name of the initializer

        Returns:
            A tensorflow initializer object instance
        """
        if name == "zeros":
            return tf.zeros_initializer()
        elif name == "uniform":
            return tf.random_uniform_initializer(minval=0, maxval=0.0001, seed=42)
        elif name == "normal":
            return tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=42)
        else:
            return tf.zeros_initializer()

    @staticmethod
    def get_parameter(name, initializer_name, shape):
        """
        Initialize a variable

        Args:
            name: the name of the variable
            initializer_name: the name of the parameter initializing method
            shape: the shape of the variable

        Returns:
            tensorflow variable
        """
        initializer = ComputerVision.get_initializer(initializer_name)
        variable = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer)
        ComputerVision.variable_summaries(variable, name)
        return variable

    @staticmethod
    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

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
        elif name == "gradientdescent":
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif name == "rmsprop":
            return tf.train.RMSPropOptimizer(learning_rate=learning_rate)
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
        Load the model variables values.

        Args:
            session: the tensorflow session

        Returns:
            Nothing
        """
        step = sorted([
            int(filename.split(self.name)[1].split("-")[1].split(".")[0])
            for filename in os.listdir(self.checkpoint_path)
            if self.name in filename
        ]).pop()
        filename = self.name + "-" + str(step)
        checkpoint_path = os.path.join(self.checkpoint_path, filename)
        self.saver.restore(session, checkpoint_path)
        self.global_step = self.global_step.assign(step)
        step = session.run(self.global_step)
        self.logger.info("Loaded model from %s at step %d" % (filename, step)
                         ) if self.logger else None

    def save(self, session, step):
        """
        Persist the model variables values.

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

    def validation_eval(self, session, summaries, dataset, step):
        pass

    def predict(self, dataset):
        pass

