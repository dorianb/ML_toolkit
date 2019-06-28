import tensorflow as tf


class Cell:

    def __init__(self):
        pass

    def build(self, input_t):
        pass

    @staticmethod
    def get_parameter(shape, initializer, name, seed=42):
        """
        Get the parameter like weight or bias.

        Args:
            shape: shape of tensor as tuple
            initializer: initializer of tensor as string
            name: name of the parameter as string
            seed: the seed for randomized initializers

        Returns:
            tensor
        """

        if initializer == "uniform":
            initializer = tf.random_uniform_initializer(seed=seed)
        elif initializer == "normal":
            initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=seed)
        elif initializer == "orthogonal":
            initializer = tf.orthogonal_initializer()
        elif initializer == "glorot_uniform":
            initializer = tf.glorot_uniform_initializer()
        elif initializer == "zeros":
            initializer = tf.zeros_initializer()
        else:
            initializer = tf.zeros_initializer()

        variable = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer)
        return variable