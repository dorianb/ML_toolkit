import tensorflow as tf


class Cell:

    def __init__(self, n_unit, return_output=True):
        pass

    def build(self):
        pass

    @staticmethod
    def initialize_variable(shape, initializer, name, seed=42):
        """

        Args:
            shape:
            initializer:
            name:
            seed:

        Returns:

        """
        if initializer == "random_uniform":
            initializer = tf.random_uniform(minval=0, seed=seed)
        elif initializer == "random_normal":
            initializer = tf.random_normal(mean=0.0, stddev=1.0, seed=seed)
        elif initializer == "orthogonal":
            initializer = tf.orthogonal_initializer(gain=1.0, seed=seed)
        elif initializer == "zeros":
            initializer = tf.zeros()
        else:
            initializer = tf.zeros()

        return tf.get_variable(name, shape=shape, initializer=initializer)