import tensorflow as tf

from computer_vision.classComputerVision import ComputerVision


class WNet(ComputerVision):
    """
    The WNet is an auto-encoder network for unsupervised semantic segmentation.
    The architecture is described onto the following paper: https://arxiv.org/abs/1711.08506

    """

    def __init__(self, batch_size=2, n_channel=3, initializer_name="zeros",
                 padding="SAME", name="WNet"):
        """
        Initialize an instance of WNet.

        Args:
            batch_size: the number of images per batch
            n_channel: the number of input channels
            initializer_name: the name of the parameters' initializer
            padding: the padding algorithm name to use
            name: the name of the object instance
        """
        tf.reset_default_graph()

        self.batch_size = batch_size
        self.n_channel = n_channel
        self.initializer_name = initializer_name
        self.padding = padding

        self.name = name

        self.input = tf.placeholder(shape=(self.batch_size, None, None, self.n_channel),
                                    dtype=tf.float32)

    def encoder_layer(self, input, initializer_name, filter_shape, filter_strides,
                      kernel_size, name):
        """
        Build the graph of an encoder layer.

        Args:
            input: input tensor of the block
            initializer_name: the name of the initializer
            filter_shape: the shape of the first convolution filter
            filter_strides: the strides of convolutions
            kernel_size: the size of the max pooling kernel
            name: the name of the layer

        Returns:
            a tensorflow operation
        """
        with tf.name_scope(name):

            conv = tf.nn.conv2d(
                input,
                filters=ComputerVision.get_parameter(
                    name="filter_1", initializer_name=initializer_name, shape=filter_shape),
                strides=filter_strides, padding=self.padding)

            conv = tf.nn.conv2d(
                conv,
                filters=ComputerVision.get_parameter(
                    name="filter_2", initializer_name=initializer_name,
                    shape=filter_shape[:2] + filter_shape[-1:] + filter_shape[-1:]),
                strides=filter_strides, padding=self.padding)

            conv = tf.nn.conv2d(
                conv,
                filters=ComputerVision.get_parameter(
                    name="filter_3", initializer_name=initializer_name,
                    shape=filter_shape[:2] + filter_shape[-1:] + filter_shape[-1:]),
                strides=filter_strides, padding=self.padding)

            return tf.nn.max_pool(conv, ksize=kernel_size, name="max_pooling")

    def build_model(self):
        """
        Build the model graph.

        Returns:
            Nothing
        """
        encoder_end_points = []
        decoder_end_points = []

        with tf.name_scope(self.name):

            with tf.name_scope("U-Enc"):

                layer_1 = self.encoder_layer(self.input, self.initializer_name,
                                             filter_shape=[3, 3, self.n_channel, 64],
                                             filter_strides=[1, 1, 1, 1],
                                             kernel_size=[1, 2, 2, 1],
                                             name="layer_1")

                layer_2 = self.encoder_layer(layer_1, self.initializer_name,
                                             filter_shape=(3, 3, 64, 128),
                                             filter_strides=[1, 1, 1, 1],
                                             kernel_size=[1, 2, 2, 1],
                                             name="layer_2")

                layer_3 = self.encoder_layer(layer_2, self.initializer_name,
                                             filter_shape=(3, 3, 128, 256),
                                             filter_strides=[1, 1, 1, 1],
                                             kernel_size=[1, 2, 2, 1],
                                             name="layer_3")

                layer_4 = self.encoder_layer(layer_3, self.initializer_name,
                                             filter_shape=(3, 3, 256, 512),
                                             filter_strides=[1, 1, 1, 1],
                                             kernel_size=[1, 2, 2, 1],
                                             name="layer_4")

                layer_5 = self.encoder_layer()

                encoder_end_points.append([layer_1, layer_2, layer_3, layer_4, layer_5])

    def fit(self):
        pass

    def predict(self):
        pass