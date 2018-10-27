import tensorflow as tf


class Cell:

    def __init__(self):
        """
        Initialize Cell object.
        """
        pass

    def build(self):
        """
        Build the Cell graph.
        """
        pass

    @staticmethod
    def initialize_variable(shape, initializer, dtype, name, seed=42):
        """
        Initialize a variable.

        Args:
            shape: the shape of variable
            initializer: the initializer to use
            dtype: type of values
            name: the scope name
            seed: the seed used by randomized methods

        Returns:
            a tensorflow variable
        """
        if initializer == "random_uniform":
            initializer = tf.random_uniform_initializer(minval=0, seed=seed)
        elif initializer == "random_normal":
            initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=seed)
        elif initializer == "orthogonal":
            initializer = tf.orthogonal_initializer(gain=1.0, seed=seed)
        elif initializer == "zeros":
            initializer = tf.zeros_initializer()
        else:
            initializer = tf.zeros_initializer()

        return tf.get_variable(name, shape=shape, dtype=dtype, initializer=initializer)