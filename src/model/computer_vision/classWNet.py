import os
from time import time
import tensorflow as tf

from computer_vision.classComputerVision import ComputerVision


class WNet(ComputerVision):
    """
    The WNet is an auto-encoder network for unsupervised semantic segmentation.
    The architecture is described onto the following paper: https://arxiv.org/abs/1711.08506

    """

    def __init__(self, batch_size=2, n_channel=3, initializer_name="zeros",
                 padding="SAME", k=2, from_pretrained=False, optimizer_name="rmsprop",
                 learning_rate=0.001, n_epochs=10, checkpoint_step=10,
                 metadata_path="", logger=None, name="WNet", debug=False):
        """
        Initialize an instance of WNet.

        Args:
            batch_size: the number of images per batch
            n_channel: the number of input channels
            initializer_name: the name of the parameters' initializer
            padding: the padding algorithm name to use
            k: the number of classes
            from_pretrained: whether to load pre-trained model
            optimizer_name: the name of the optimizer
            learning_rate: the learning rate
            n_epochs: the number of epochs
            checkpoint_step: the number of training step between checkpoints
            metadata_path: the folder path where to save metadata
            logger: a logging instance object
            name: the name of the object instance
            debug: whether to activate the debug mode
        """
        ComputerVision.__init__(summary_path="", checkpoint_path="")

        self.batch_size = batch_size
        self.n_channel = n_channel
        self.initializer_name = initializer_name
        self.padding = padding
        self.k = k
        self.n_epochs = n_epochs
        self.checkpoint_step = checkpoint_step
        self.summary_path = os.path.join(metadata_path, "summaries", name)
        self.checkpoint_path = os.path.join(metadata_path, "checkpoints", name)
        self.name = name
        self.from_pretrained = from_pretrained
        self.logger = logger
        self.debug = debug

        # Input
        self.input = tf.placeholder(shape=(self.batch_size, None, None, self.n_channel),
                                    dtype=tf.float32)

        # Build model
        self.segmentation, self.reconstruction = self.build_model()

        # Global step
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")
        self.global_step = tf.add(self.global_step, tf.constant(1))

        # Optimizer
        self.optimizer = ComputerVision.get_optimizer(optimizer_name, learning_rate)

        # Summary writers
        self.train_writer, self.validation_writer = ComputerVision.get_writer(self)

        # Model saver
        self.saver = tf.train.Saver()

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

    def fit(self, training_set, validation_set):
        """
        Fit the model weights with input and labels.

        Args:
            training_set: the training input and label
            validation_set: the validation input and label

        Returns: Nothing
        """
        # Loss
        n_cut_loss = ComputerVision.compute_soft_ncut(self.segmentation, self.k)
        reconstruction_loss = ComputerVision.compute_loss(self.input, self.reconstruction, loss_name="mse")

        # Accuracy
        accuracy = ComputerVision.compute_accuracy(self, self.input, self.reconstruction)

        # Optimization
        train_op_1 = ComputerVision.compute_gradient(self, n_cut_loss, self.global_step)
        train_op_2 = ComputerVision.compute_gradient(self, reconstruction_loss, self.global_step)

        # Merge summaries
        summaries = tf.summary.merge_all()

        # Initialize variables
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        with tf.Session() as sess:

            sess.run(init_g)
            sess.run(init_l)

            self.train_writer.add_graph(sess.graph)

            # Load existing model
            ComputerVision.load(self, sess) if self.from_pretrained else None

            for epoch in range(self.n_epochs):

                for i in range(self.batch_size, len(training_set), self.batch_size):

                    time0 = time()

                    # Load batch
                    batch_examples = training_set[i - self.batch_size: i]
                    image_batch, label_batch = self.load_batch(batch_examples)

                    # Update U-Enc
                    _, n_cut_loss_value = sess.run([
                        train_op_1, n_cut_loss],
                        feed_dict={
                            self.input: image_batch
                        }
                    )

                    # Update U-Enc and U-Dec
                    _, reconstruction_loss_value, summaries_value, accuracy_value, step = sess.run(
                        [train_op_2, reconstruction_loss, summaries, accuracy, self.global_step],
                        feed_dict={
                            self.input: image_batch
                        }
                    )

                    self.logger.info("Writing summary to {0}".format(self.summary_path)) if self.logger else None
                    self.train_writer.add_summary(summaries_value, step)

                    time1 = time()
                    self.logger.info(
                        "Accuracy = {0}, N-Cut Loss = {1}, Reconstruction Loss = {2} for batch {3} in {4:.2f} seconds".format(
                            accuracy_value, n_cut_loss_value, reconstruction_loss_value, i / self.batch_size, time1 - time0)) if self.logger else None

                    if i % self.checkpoint_step == 0:

                        ComputerVision.save(self, sess, step=self.global_step)

    def predict(self, data_set):
        pass