import tensorflow as tf

from computer_vision.classComputerVision import ComputerVision


class WNet(ComputerVision):
    """
    The WNet is an auto-encoder network for unsupervised semantic segmentation.
    The architecture is described onto the following paper: https://arxiv.org/abs/1711.08506

    """

    def __init__(self, batch_size=2, n_channel=3, initializer_name="zeros",
                 padding="SAME", k=2, name="WNet"):
        """
        Initialize an instance of WNet.

        Args:
            batch_size: the number of images per batch
            n_channel: the number of input channels
            initializer_name: the name of the parameters' initializer
            padding: the padding algorithm name to use
            k: the number of classes
            name: the name of the object instance
        """
        tf.reset_default_graph()

        self.batch_size = batch_size
        self.n_channel = n_channel
        self.initializer_name = initializer_name
        self.padding = padding
        self.k = k

        self.name = name

        self.input = tf.placeholder(shape=(self.batch_size, None, None, self.n_channel),
                                    dtype=tf.float32)

    def layer(self, input, initializer_name, filter_shape, filter_strides,
              kernel_size, separable_conv=False, training=True, link="max_pooling",
              skip_input=None, name=""):
        """
        Build the graph for a layer.

        Args:
            input: input tensor of the block
            initializer_name: the name of the initializer
            filter_shape: the shape of the first convolution filter
            filter_strides: the strides of convolutions
            kernel_size: the size of the max pooling kernel
            separable_conv: whether to use a separable convolution
            training: whether to update the parameters
            link: the name of the operation applied to layer output ('max_pooling' or 'up_conv')
            skip_input: the output of same layer
            name: the name of the layer

        Returns:
            a tensorflow operation
        """
        with tf.variable_scope(name):

            channel_multiplier = 1
            conv = tf.concat([input, skip_input], axis=-1) if skip_input else input

            if separable_conv:

                conv = tf.nn.separable_conv2d(
                    conv,
                    depthwise_filter=ComputerVision.get_parameter(
                        name="filter_1_1", initializer_name=initializer_name,
                        shape=filter_shape[:3] + [channel_multiplier]
                    ),
                    pointwise_filter=ComputerVision.get_parameter(
                        name="filter_1_2", initializer_name=initializer_name,
                        shape=[1, 1] + [filter_shape[2]*channel_multiplier] + filter_shape[-1:]
                    ),
                    strides=filter_strides,
                    padding=self.padding
                )

            else:

                conv = tf.nn.conv2d(
                    conv,
                    filters=ComputerVision.get_parameter(
                        name="filter_1", initializer_name=initializer_name,
                        shape=filter_shape
                    ),
                    strides=filter_strides, padding=self.padding)

            conv = tf.nn.relu(conv)
            conv = tf.layers.batch_normalization(conv, training=training)

            if separable_conv:

                conv = tf.nn.separable_conv2d(
                    conv,
                    depthwise_filter=ComputerVision.get_parameter(
                        name="filter_2_1", initializer_name=initializer_name,
                        shape=filter_shape[:2] + filter_shape[-1:] + [channel_multiplier]
                    ),
                    pointwise_filter=ComputerVision.get_parameter(
                        name="filter_2_2", initializer_name=initializer_name,
                        shape=[1, 1] + [filter_shape[-1]*channel_multiplier] + filter_shape[-1:]
                    ),
                    strides=filter_strides,
                    padding=self.padding
                )

            else:

                conv = tf.nn.conv2d(
                    conv,
                    filters=ComputerVision.get_parameter(
                        name="filter_2_1", initializer_name=initializer_name,
                        shape=filter_shape[:2] + filter_shape[-1:] + filter_shape[-1:]
                    ),
                    strides=filter_strides, padding=self.padding)

            conv = tf.nn.relu(conv)
            conv = tf.layers.batch_normalization(conv, training=training)

            if link == "max_pooling":
                output = tf.nn.max_pool(conv, ksize=kernel_size, name=link)

            elif link == "up_conv":
                output = tf.nn.conv2d_transpose(
                    conv,
                    filter=ComputerVision.get_parameter(
                        name=link, initializer_name=initializer_name,
                        shape=kernel_size[1:3] + filter_shape[-1:] / 2 + filter_shape[-1:]
                    ),
                    output_shape=tf.concat([tf.shape(conv)[0], tf.shape(conv)[1:3] * 2, filter_shape[-1:] / 2], 0),
                    strides=filter_strides, padding=self.padding
                )

            elif link == "conv_soft":
                conv = tf.nn.conv2d(
                    conv,
                    filters=ComputerVision.get_parameter(
                        name=link, initializer_name=initializer_name,
                        shape=[1, 1] + filter_shape[-1:] + [self.k]
                    ),
                    strides=filter_strides, padding=self.padding)
                output = tf.nn.softmax(conv, axis=-1)

            elif link == "conv":
                output = tf.nn.conv2d(
                    conv,
                    filters=ComputerVision.get_parameter(
                        name=link, initializer_name=initializer_name,
                        shape=[1, 1] + filter_shape[-1:] + [self.n_channel]
                    ),
                    strides=filter_strides, padding=self.padding)

            else:
                output = conv

            return output, conv

    def build_model(self):
        """
        Build the model graph.

        Returns:
            a tuple of two tensors representing the segmented and the reconstructed image
        """

        with tf.variable_scope(self.name):

            with tf.variable_scope("U_Enc"):

                with tf.variable_scope("Contractive_path"):

                    l1_out, l1_conv = self.layer(self.input, self.initializer_name,
                                         filter_shape=[3, 3, self.n_channel, 64],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=False,
                                         training=True,
                                         link="max_pooling",
                                         name="layer_1")

                    l2_out, l2_conv = self.layer(l1_out, self.initializer_name,
                                         filter_shape=[3, 3, 64, 128],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="max_pooling",
                                         name="layer_2")

                    l3_out, l3_conv = self.layer(l2_out, self.initializer_name,
                                         filter_shape=[3, 3, 128, 256],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="max_pooling",
                                         name="layer_3")

                    l4_out, l4_conv = self.layer(l3_out, self.initializer_name,
                                         filter_shape=[3, 3, 256, 512],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="max_pooling",
                                         name="layer_4")

                    l5_out, l5_conv = self.layer(l4_out, self.initializer_name,
                                         filter_shape=[3, 3, 512, 1024],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="up_conv",
                                         name="layer_5")

                with tf.name_scope("Expansive_path"):

                    l6_out, _ = self.layer(l5_out, self.initializer_name,
                                         filter_shape=[3, 3, 1024, 512],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=False,
                                         training=True,
                                         link="up_conv",
                                         skip_input=l4_conv,
                                         name="layer_6")

                    l7_out, _ = self.layer(l6_out, self.initializer_name,
                                         filter_shape=[3, 3, 512, 256],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="up_conv",
                                         skip_input=l3_conv,
                                         name="layer_7")

                    l8_out, _ = self.layer(l7_out, self.initializer_name,
                                         filter_shape=[3, 3, 256, 128],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="up_conv",
                                         skip_input=l2_conv,
                                         name="layer_8")

                    l9_out, _ = self.layer(l8_out, self.initializer_name,
                                         filter_shape=[3, 3, 128, 64],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=False,
                                         training=True,
                                         link="conv_soft",
                                         skip_input=l1_conv,
                                         name="layer_9")

            with tf.variable_scope("U_Dec"):

                with tf.variable_scope("Contractive_path"):

                    l10_out, l10_conv = self.layer(l9_out, self.initializer_name,
                                         filter_shape=[3, 3, self.k, 64],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=False,
                                         training=True,
                                         link="max_pooling",
                                         name="layer_10")

                    l11_out, l11_conv = self.layer(l10_out, self.initializer_name,
                                         filter_shape=[3, 3, 64, 128],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="max_pooling",
                                         name="layer_11")

                    l12_out, l12_conv = self.layer(l11_out, self.initializer_name,
                                         filter_shape=[3, 3, 128, 256],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="max_pooling",
                                         name="layer_12")

                    l13_out, l13_conv = self.layer(l12_out, self.initializer_name,
                                         filter_shape=[3, 3, 256, 512],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="max_pooling",
                                         name="layer_13")

                    l14_out, l14_conv = self.layer(l13_out, self.initializer_name,
                                         filter_shape=[3, 3, 512, 1024],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="up_conv",
                                         name="layer_14")

                with tf.name_scope("Expansive_path"):

                    l15_out, _ = self.layer(l14_out, self.initializer_name,
                                         filter_shape=[3, 3, 1024, 512],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=False,
                                         training=True,
                                         link="up_conv",
                                         skip_input=l13_conv,
                                         name="layer_15")

                    l16_out, _ = self.layer(l15_out, self.initializer_name,
                                         filter_shape=[3, 3, 512, 256],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="up_conv",
                                         skip_input=l12_conv,
                                         name="layer_16")

                    l17_out, _ = self.layer(l16_out, self.initializer_name,
                                         filter_shape=[3, 3, 256, 128],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=True,
                                         training=True,
                                         link="up_conv",
                                         skip_input=l11_conv,
                                         name="layer_17")

                    l18_out, _ = self.layer(l17_out, self.initializer_name,
                                         filter_shape=[3, 3, 128, 64],
                                         filter_strides=[1, 1, 1, 1],
                                         kernel_size=[1, 2, 2, 1],
                                         separable_conv=False,
                                         training=True,
                                         link="conv",
                                         skip_input=l10_conv,
                                         name="layer_18")

        return l9_out, l18_out

    def fit(self):
        pass

    def predict(self):
        pass