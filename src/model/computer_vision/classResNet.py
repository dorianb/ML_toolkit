import tensorflow as tf
import os

from computer_vision.classImageClassification import ImageClassification


class ResNet(ImageClassification):

    """
    The Resnet implementation is based on the following paper: https://arxiv.org/abs/1512.03385
    """

    def __init__(self, classes, n_layers=34, batch_size=1, height=224, width=224, dim_out=10,
                 grayscale=True, binarize=True, normalize=False,
                 learning_rate=10, n_epochs=1, validation_step=10,
                 checkpoint_step=100, is_encoder=True, validation_size=10,
                 optimizer="adam", metadata_path="", name="vgg",
                 from_pretrained=False, is_training=True, logger=None, debug=False):
        """
        Initialization of the Vgg model.

        Args:
            classes: dictionary of classes
            n_layers: number of layers in the nework
            batch_size: the size of batch
            height: the height of the image
            width: the width of the image
            grayscale: whether the input image are grayscale
            binarize: whether input image are binarized
            normalize: whether input image are normalized
            dim_out: the output dimension of the model
            learning_rate: the learning rate applied in the gradient descent optimization
            n_epochs: the number of epochs
            validation_step: the number of training examples to use for training before
                evaluation on validation dataset
            checkpoint_step: the number of batch training examples to use before
                checkpoint
            is_encoder: the vgg is used as an encoder
            validation_size: the number of examples to use for validation
            optimizer: the optimizer to use
            metadata_path: the path to the metadata
            name: the name of the object instance
            from_pretrained: whether to start training from pre-trained model
            is_training: whether the model is used for training or prediction
            logger: an instance object of logging module.
            debug: whether the debug mode is activated or not

        Returns:
            Nothing
        """
        tf.reset_default_graph()

        ImageClassification.__init__(self)

        self.n_layers = n_layers
        self.batch_size = batch_size
        self.grayscale = grayscale
        self.binarize = binarize
        self.normalize = normalize
        self.n_channel = 1 if self.grayscale else 3
        self.resize_dim = (width, height)
        self.dim_out = dim_out
        self.classes = classes
        self.n_classes = len(classes)
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.validation_step = validation_step
        self.checkpoint_step = checkpoint_step
        self.is_encoder = is_encoder
        self.validation_size = validation_size
        self.optimizer_name = optimizer
        self.summary_path = os.path.join(metadata_path, "summaries", name)
        self.checkpoint_path = os.path.join(metadata_path, "checkpoints", name)
        self.name = name
        self.from_pretrained = from_pretrained
        self.logger = logger
        self.debug = debug

        # Input
        self.input = tf.placeholder(shape=(None, None, None, self.n_channel), dtype=tf.float32)

        # Label
        self.label = tf.placeholder(shape=(None, self.n_classes), dtype=tf.float32)
        self.label = tf.Print(self.label, [self.label], message="Truth label: ",
                              summarize=self.n_classes * self.batch_size) if self.debug else self.label

        # Build model
        self.model = self.build_model(self.dim_out, is_training=is_training)

        # Global step
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Optimizer
        self.optimizer = ImageClassification.get_optimizer(self.optimizer_name, self.learning_rate)

        # Summary writers
        self.train_writer, self.validation_writer = ImageClassification.get_writer(self)

        # Model saver
        self.saver = tf.train.Saver()

    def residual_block_2(self, input, n_channels=64, n_filters=64, padding="SAME",
                         is_training=True, with_stride=[1, 1, 1, 1], scope="conv_x"):
        """
        Build a residual block with two convolution layers.

        Args:
            input: tensorflow operation
            n_channels: number of input channels as integer
            n_filters: number of filters as integer
            padding: the padding policy used by convolutions
            is_training: whether to use training layers
            with_stride: the stride to apply for the first convolution as an integer
            scope: the name of scope

        Returns:
            tensorflow operation
        """
        with tf.variable_scope(scope, values=[input]):
            net = tf.nn.conv2d(
                    input,
                    filter=ImageClassification.get_parameter("1", "xavier", [3, 3, n_channels, n_filters]),
                    strides=with_stride, padding=padding)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            net = tf.nn.conv2d(
                net,
                filter=ImageClassification.get_parameter("2", "xavier", [3, 3, n_filters, n_filters]),
                strides=[1, 1, 1, 1], padding=padding)
            net = tf.layers.batch_normalization(net, training=is_training)

            if with_stride == [1, 2, 2, 1]:
                input = tf.nn.conv2d(
                        input,
                        filter=ImageClassification.get_parameter("3", "xavier", [1, 1, n_channels, n_channels]),
                        strides=with_stride, padding=padding)

            return tf.nn.relu(tf.add(net, input))

    def residual_block_3(self, input, n_channels=256, n_filters=64, padding="SAME",
                         is_training=True, with_stride=[1, 1, 1, 1], scope="conv_x"):
        """
        Build a residual block with three convolution layers.

        Args:
            input: tensorflow operation
            n_channels: number of input channels as integer
            n_filters: number of filters as integer
            padding: the padding policy used by convolutions
            is_training: whether to use training layers
            with_stride: the stride to apply for the first convolution as an integer
            scope: the name of scope

        Returns:
            tensorflow operation
        """
        with tf.variable_scope(scope, values=[input]):
            net = tf.nn.conv2d(
                    input,
                    filter=ImageClassification.get_parameter("1", "xavier", [1, 1, n_channels, n_filters]),
                    strides=with_stride, padding=padding)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            net = tf.nn.conv2d(
                    net,
                    filter=ImageClassification.get_parameter("2", "xavier", [3, 3, n_filters, n_filters]),
                    strides=[1, 1, 1, 1], padding=padding)
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)
            net = tf.nn.conv2d(
                net,
                filter=ImageClassification.get_parameter("3", "xavier", [1, 1, n_filters, n_channels]),
                strides=[1, 1, 1, 1], padding=padding)
            net = tf.layers.batch_normalization(net, training=is_training)

            if with_stride == [1, 2, 2, 1]:
                input = tf.nn.conv2d(
                    input,
                    filter=ImageClassification.get_parameter("4", "xavier", [1, 1, n_channels, n_channels]),
                    strides=with_stride, padding=padding)

            return tf.nn.relu(tf.add(net, input))

    def shortcut(self, input, dim_in=64, dim_out=128, mode="pad", scope=""):
        """
        Create a shortcut between two resnet block.

        Args:
            input: tensorflow operation
            dim_in: the number of channels in input tensor
            dim_out: the number of channels in output tensor
            mode: mode of shortcut ("pad" or "conv")

        Returns:
            tensorflow operation
        """
        if mode == "pad":
            extra_zeros = dim_out - dim_in
            left_zeros = int(extra_zeros / 2)
            right_zeros = extra_zeros - left_zeros
            padding = tf.constant([[0, 0], [0, 0], [0, 0], [left_zeros, right_zeros]])
            return tf.pad(input, padding, constant_values=0)
        elif mode == "conv":
            pass

    def build_model(self, dim_output=1000, is_training=True, fc_padding='SAME',
                    shortcut_mode="pad", name=None):
        """
        Build the ResNet graph model.

        Args:
            dim_output: the dimension output of the network
            is_training: whether to use training layers
            fc_padding: the padding used for the fully connected layers
            shortcut_mode: the shortcut mode to apply ("pad" or "conv")
            name: the name of the graph operations

        Returns:
            the last layer of the model
        """
        with tf.variable_scope(name, 'ResNet-'+str(self.n_layers), [self.input]) as sc:

            net = tf.nn.relu(tf.nn.conv2d(
                self.input,
                filter=ImageClassification.get_parameter("conv_1", "xavier", [7, 7, self.n_channel, 64]),
                strides=[1, 2, 2, 1], padding="SAME", name="conv_1")
            )

            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="conv_2")

            if self.n_layers == 18:
                # 2 x 64
                net = self.residual_block_2(net, n_channels=64, n_filters=64, is_training=is_training, scope="conv_2_1")
                net = self.residual_block_2(net, n_channels=64, n_filters=64, is_training=is_training, scope="conv_2_2")
                # 2 x 128
                net = self.shortcut(net, dim_in=64, dim_out=128, mode=shortcut_mode)
                net = self.residual_block_2(net, n_channels=128, n_filters=128, is_training=is_training, with_stride=[1, 2, 2, 1], scope="conv_3_1")
                net = self.residual_block_2(net, n_channels=128, n_filters=128, is_training=is_training, scope="conv_3_2")
                # 2 x 256
                net = self.shortcut(net, dim_in=128, dim_out=256, mode=shortcut_mode)
                net = self.residual_block_2(net, n_channels=256, n_filters=256, is_training=is_training, with_stride=[1, 2, 2, 1], scope="conv_4_1")
                net = self.residual_block_2(net, n_channels=256, n_filters=256, is_training=is_training, scope="conv_4_2")
                # 2 x 512
                net = self.shortcut(net, dim_in=256, dim_out=512, mode=shortcut_mode)
                net = self.residual_block_2(net, n_channels=512, n_filters=512, is_training=is_training, with_stride=[1, 2, 2, 1], scope="conv_5_1")
                net = self.residual_block_2(net, n_channels=512, n_filters=512, is_training=is_training, scope="conv_5_2")
                n_channels = 512

            elif self.n_layers == 34:
                # 3 x 64
                net = self.residual_block_2(net, n_channels=64, n_filters=64, is_training=is_training, scope="conv_2_1")
                net = self.residual_block_2(net, n_channels=64, n_filters=64, is_training=is_training, scope="conv_2_2")
                net = self.residual_block_2(net, n_channels=64, n_filters=64, is_training=is_training, scope="conv_2_3")
                # 4 x 128
                net = self.shortcut(net, dim_in=64, dim_out=128, mode=shortcut_mode)
                net = self.residual_block_2(net, n_channels=128 ,n_filters=128, is_training=is_training, with_stride=[1, 2, 2, 1], scope="conv_3")
                net = self.residual_block_2(net, n_channels=128, n_filters=128, is_training=is_training, scope="conv_3_1")
                net = self.residual_block_2(net, n_channels=128, n_filters=128, is_training=is_training, scope="conv_3_2")
                net = self.residual_block_2(net, n_channels=128, n_filters=128, is_training=is_training, scope="conv_3_3")
                # 6 x 256
                net = self.shortcut(net, dim_in=128, dim_out=256, mode=shortcut_mode)
                net = self.residual_block_2(net, n_channels=256, n_filters=256, is_training=is_training, with_stride=[1, 2, 2, 1], scope="conv_4_1")
                net = self.residual_block_2(net, n_channels=256, n_filters=256, is_training=is_training, scope="conv_4_2")
                net = self.residual_block_2(net, n_channels=256, n_filters=256, is_training=is_training, scope="conv_4_3")
                net = self.residual_block_2(net, n_channels=256, n_filters=256, is_training=is_training, scope="conv_4_4")
                net = self.residual_block_2(net, n_channels=256, n_filters=256, is_training=is_training, scope="conv_4_5")
                net = self.residual_block_2(net, n_channels=256, n_filters=256, is_training=is_training, scope="conv_4_6")
                # 3 x 512
                net = self.shortcut(net, dim_in=256, dim_out=512, mode=shortcut_mode)
                net = self.residual_block_2(net, n_channels=512, n_filters=512, is_training=is_training, with_stride=[1, 2, 2, 1], scope="conv_5_1")
                net = self.residual_block_2(net, n_channels=512, n_filters=512, is_training=is_training, scope="conv_5_2")
                net = self.residual_block_2(net, n_channels=512, n_filters=512, is_training=is_training, scope="conv_5_3")
                n_channels = 512

            elif self.n_layers == 50:
                # 3 x 64
                net = self.shortcut(net, dim_in=64, dim_out=256, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=256, n_filters=64, is_training=is_training, scope="conv_2_1")
                net = self.residual_block_3(net, n_channels=256, n_filters=64, is_training=is_training, scope="conv_2_2")
                net = self.residual_block_3(net, n_channels=256, n_filters=64, is_training=is_training, scope="conv_2_3")
                # 4 x 128
                net = self.shortcut(net, dim_in=256, dim_out=512, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training, with_stride=[1, 2, 2, 1], scope="conv_3_1")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training, scope="conv_3_2")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training, scope="conv_3_3")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training, scope="conv_3_4")
                # 6 x 256
                net = self.shortcut(net, dim_in=512, dim_out=1024, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training, with_stride=[1, 2, 2, 1], scope="conv_4_1")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training, scope="conv_4_2")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training, scope="conv_4_3")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training, scope="conv_4_4")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training, scope="conv_4_5")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training, scope="conv_4_6")
                # 3 x 512
                net = self.shortcut(net, dim_in=1024, dim_out=2048, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=2048, n_filters=512, is_training=is_training, with_stride=[1, 2, 2, 1], scope="conv_5_1")
                net = self.residual_block_3(net, n_channels=2048, n_filters=512, is_training=is_training, scope="conv_5_2")
                net = self.residual_block_3(net, n_channels=2048, n_filters=512, is_training=is_training, scope="conv_5_3")
                n_channels = 2048

            elif self.n_layers == 101:
                # 3 x 64
                net = self.shortcut(net, dim_in=64, dim_out=256, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=256, n_filters=64, is_training=is_training,
                                            scope="conv_2_1")
                net = self.residual_block_3(net, n_channels=256, n_filters=64, is_training=is_training,
                                            scope="conv_2_2")
                net = self.residual_block_3(net, n_channels=256, n_filters=64, is_training=is_training,
                                            scope="conv_2_3")
                # 4 x 128
                net = self.shortcut(net, dim_in=256, dim_out=512, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            with_stride=[1, 2, 2, 1], scope="conv_3_1")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_2")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_3")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_4")
                # 23 x 256
                net = self.shortcut(net, dim_in=512, dim_out=1024, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            with_stride=[1, 2, 2, 1], scope="conv_4_1")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_2")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_3")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_4")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_5")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_6")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_7")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_8")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_9")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_10")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_11")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_12")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_13")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_14")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_15")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_16")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_17")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_18")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_19")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_20")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_21")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_22")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_23")
                # 3 x 512
                net = self.shortcut(net, dim_in=1024, dim_out=2048, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=2048, n_filters=512, is_training=is_training,
                                            with_stride=[1, 2, 2, 1], scope="conv_5_1")
                net = self.residual_block_3(net, n_channels=2048, n_filters=512, is_training=is_training,
                                            scope="conv_5_2")
                net = self.residual_block_3(net, n_channels=2048, n_filters=512, is_training=is_training,
                                            scope="conv_5_3")
                n_channels = 2048

            elif self.n_layers == 152:
                # 3 x 64
                net = self.shortcut(net, dim_in=64, dim_out=256, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=256, n_filters=64, is_training=is_training,
                                            scope="conv_2_1")
                net = self.residual_block_3(net, n_channels=256, n_filters=64, is_training=is_training,
                                            scope="conv_2_2")
                net = self.residual_block_3(net, n_channels=256, n_filters=64, is_training=is_training,
                                            scope="conv_2_3")
                # 8 x 128
                net = self.shortcut(net, dim_in=256, dim_out=512, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            with_stride=[1, 2, 2, 1], scope="conv_3_1")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_2")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_3")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_4")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_5")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_6")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_7")
                net = self.residual_block_3(net, n_channels=512, n_filters=128, is_training=is_training,
                                            scope="conv_3_8")
                # 36 x 256
                net = self.shortcut(net, dim_in=512, dim_out=1024, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            with_stride=[1, 2, 2, 1], scope="conv_4_1")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_2")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_3")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_4")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_5")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_6")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_7")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_8")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_9")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_10")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_11")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_12")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_13")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_14")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_15")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_16")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_17")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_18")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_19")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_20")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_21")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_22")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_23")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_24")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_25")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_26")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_27")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_28")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_29")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_30")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_31")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_32")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_33")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_34")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_35")
                net = self.residual_block_3(net, n_channels=1024, n_filters=256, is_training=is_training,
                                            scope="conv_4_36")
                # 3 x 512
                net = self.shortcut(net, dim_in=1024, dim_out=2048, mode=shortcut_mode)
                net = self.residual_block_3(net, n_channels=2048, n_filters=512, is_training=is_training,
                                            with_stride=[1, 2, 2, 1], scope="conv_5_1")
                net = self.residual_block_3(net, n_channels=2048, n_filters=512, is_training=is_training,
                                            scope="conv_5_2")
                net = self.residual_block_3(net, n_channels=2048, n_filters=512, is_training=is_training,
                                            scope="conv_5_3")
                n_channels = 2048

            net = tf.nn.avg_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            if self.n_classes:
                net = tf.nn.dropout(net, 0.5, name='dropout6')
                net = tf.nn.conv2d(net,
                       filter=ImageClassification.get_parameter("fc6", "xavier", [1, 1, n_channels, self.n_classes]),
                       strides=[1, 1, 1, 1], padding=fc_padding, name='fc6')
                net = tf.squeeze(net, [1, 2], name='fc6/squeezed')
            return net
