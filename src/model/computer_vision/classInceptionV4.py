import tensorflow as tf
import os

from computer_vision.classImageClassification import ImageClassification


class InceptionV4(ImageClassification):

    """
    The Inception implementation is based on the following paper: https://arxiv.org/abs/1602.07261
    """

    def __init__(self, classes, batch_size=1, height=299, width=299, dim_out=10,
                 grayscale=True, binarize=True, normalize=False,
                 learning_rate=10, n_epochs=1, validation_step=10,
                 checkpoint_step=100, is_encoder=True, validation_size=10,
                 optimizer="adam", metadata_path="", name="vgg",
                 from_pretrained=False, is_training=True, logger=None, debug=False):
        """
        Initialization of the Inception model.

        Args:
            classes: dictionary of classes
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

    def inception_C(self, input, n_channels=1536, is_training=True, scope="InceptionC"):
        """
        Build inception C

        Args:
            input: tensorflow operation
            n_channels: number of input channels
            is_training: whether to use training operations
            scope: name of variable scope
        Returns:
            tensorflow operation

        """
        with tf.variable_scope(scope, values=[input]):

            part1 = tf.nn.avg_pool(input, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME")
            part1 = tf.nn.conv2d(
                part1,
                filter=ImageClassification.get_parameter("1", "xavier", [1, 1, n_channels, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part1 = tf.layers.batch_normalization(part1, training=is_training)
            part1 = tf.nn.relu(part1)

            part2 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("2", "xavier", [1, 1, n_channels, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part2 = tf.layers.batch_normalization(part2, training=is_training)
            part2 = tf.nn.relu(part2)

            part3 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("3", "xavier", [1, 1, n_channels, 384]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)

            part31 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("4", "xavier", [1, 3, 384, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part31 = tf.layers.batch_normalization(part31, training=is_training)
            part31 = tf.nn.relu(part31)

            part32 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("5", "xavier", [3, 1, 384, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part32 = tf.layers.batch_normalization(part32, training=is_training)
            part32 = tf.nn.relu(part32)

            part4 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("6", "xavier", [1, 1, n_channels, 384]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)
            part4 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("7", "xavier", [1, 3, 384, 448]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)
            part4 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("8", "xavier", [3, 1, 448, 512]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)

            part41 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("9", "xavier", [3, 1, 512, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part41 = tf.layers.batch_normalization(part41, training=is_training)
            part41 = tf.nn.relu(part41)

            part42 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("10", "xavier", [1, 3, 512, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part42 = tf.layers.batch_normalization(part42, training=is_training)
            part42 = tf.nn.relu(part42)

            return tf.concat([part1, part2, part31, part32, part41, part42], axis=-1)

    def reduction_B(self, input, n_channels=384, is_training=True, scope="ReductionB"):
        """
        Build the reduction B.

        Args:
            input: tensorflow operation
            n_channels: number of input channels
            is_training: whether to use training operations
            scope: name of variable scope
        Returns:
            tensorflow operation:

        """
        with tf.variable_scope(scope, values=[input]):
            part1 = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

            part2 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("1", "xavier", [1, 1, n_channels, 192]),
                strides=[1, 1, 1, 1], padding="SAME")
            part2 = tf.layers.batch_normalization(part2, training=is_training)
            part2 = tf.nn.relu(part2)
            part2 = tf.nn.conv2d(
                part2,
                filter=ImageClassification.get_parameter("2", "xavier", [3, 3, 192, 192]),
                strides=[1, 2, 2, 1], padding="VALID")
            part2 = tf.layers.batch_normalization(part2, training=is_training)
            part2 = tf.nn.relu(part2)

            part3 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("3", "xavier", [1, 1, n_channels, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)
            part3 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("4", "xavier", [1, 7, 256, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)
            part3 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("5", "xavier", [7, 1, 256, 320]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)
            part3 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("6", "xavier", [3, 3, 320, 320]),
                strides=[1, 2, 2, 1], padding="VALID")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)

            return tf.concat([part1, part2, part3], axis=-1)

    def inception_B(self, input, n_channels=1024, is_training=True, scope="InceptionB"):
        """
        Build inception B

        Args:
            input: tensorflow operation
            n_channels: number of input channels
            is_training: whether to use training operations
            scope: name of variable scope
        Returns:
            tensorflow operation

        """
        with tf.variable_scope(scope, values=[input]):

            part1 = tf.nn.avg_pool(input, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME")
            part1 = tf.nn.conv2d(
                part1,
                filter=ImageClassification.get_parameter("1", "xavier", [1, 1, n_channels, 128]),
                strides=[1, 1, 1, 1], padding="SAME")
            part1 = tf.layers.batch_normalization(part1, training=is_training)
            part1 = tf.nn.relu(part1)

            part2 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("2", "xavier", [1, 1, n_channels, 384]),
                strides=[1, 1, 1, 1], padding="SAME")
            part2 = tf.layers.batch_normalization(part2, training=is_training)
            part2 = tf.nn.relu(part2)

            part3 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("3", "xavier", [1, 1, n_channels, 192]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)
            part3 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("4", "xavier", [1, 7, 192, 224]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)
            part3 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("5", "xavier", [1, 7, 224, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)

            part4 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("6", "xavier", [1, 1, n_channels, 192]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)
            part4 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("7", "xavier", [1, 7, 192, 192]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)
            part4 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("8", "xavier", [7, 1, 192, 224]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)
            part4 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("9", "xavier", [1, 7, 224, 224]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)
            part4 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("10", "xavier", [7, 1, 224, 256]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)

            return tf.concat([part1, part2, part3, part4], axis=-1)

    def reduction_A(self, input, klmn=(192, 224, 256, 384), n_channels=384, is_training=True, scope="ReductionA"):
        """
        Build the reduction A.

        Args:
            input: tensorflow operation
            klmn: a list of filters
            n_channels: number of input channels
            is_training: whether to use training operations
            scope: name of variable scope
        Returns:
            tensorflow operation:

        """
        k, l, m, n = klmn

        with tf.variable_scope(scope, values=[input]):
            part1 = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

            part2 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("1", "xavier", [3, 3, n_channels, n]),
                strides=[1, 2, 2, 1], padding="VALID")
            part2 = tf.layers.batch_normalization(part2, training=is_training)
            part2 = tf.nn.relu(part2)

            part3 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("2", "xavier", [1, 1, n_channels, k]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)
            part3 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("3", "xavier", [3, 3, k, l]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)
            part3 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("4", "xavier", [3, 3, l, m]),
                strides=[1, 2, 2, 1], padding="VALID")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)

            return tf.concat([part1, part2, part3], axis=-1)

    def inception_A(self, input, n_channels=384, is_training=True, scope="InceptionA"):
        """
        Build inception A

        Args:
            input: tensorflow operation
            n_channels: number of input channels
            is_training: whether to use training operations
            scope: name of variable scope
        Returns:
            tensorflow operation

        """
        with tf.variable_scope(scope, values=[input]):

            part1 = tf.nn.avg_pool(input, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME")
            part1 = tf.nn.conv2d(
                part1,
                filter=ImageClassification.get_parameter("1", "xavier", [1, 1, n_channels, 96]),
                strides=[1, 1, 1, 1], padding="SAME")
            part1 = tf.layers.batch_normalization(part1, training=is_training)
            part1 = tf.nn.relu(part1)

            part2 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("2", "xavier", [1, 1, n_channels, 96]),
                strides=[1, 1, 1, 1], padding="SAME")
            part2 = tf.layers.batch_normalization(part2, training=is_training)
            part2 = tf.nn.relu(part2)

            part3 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("3", "xavier", [1, 1, n_channels, 64]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)
            part3 = tf.nn.conv2d(
                part3,
                filter=ImageClassification.get_parameter("4", "xavier", [3, 3, 64, 96]),
                strides=[1, 1, 1, 1], padding="SAME")
            part3 = tf.layers.batch_normalization(part3, training=is_training)
            part3 = tf.nn.relu(part3)

            part4 = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("5", "xavier", [1, 1, n_channels, 64]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)
            part4 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("6", "xavier", [3, 3, 64, 96]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)
            part4 = tf.nn.conv2d(
                part4,
                filter=ImageClassification.get_parameter("7", "xavier", [3, 3, 96, 96]),
                strides=[1, 1, 1, 1], padding="SAME")
            part4 = tf.layers.batch_normalization(part4, training=is_training)
            part4 = tf.nn.relu(part4)

            return tf.concat([part1, part2, part3, part4], axis=-1)

    def stem(self, input, n_channels=3, is_training=True, scope="stem"):
        """
        Build stem

        Args:
            input: tensorflow operation
            n_channels: number of input channels
            is_training: whether to use training operations
            scope: name of variable scope
        Returns:
            tensorflow operation
        """
        with tf.variable_scope(scope, values=[input]):
            # First part
            net = tf.nn.conv2d(
                input,
                filter=ImageClassification.get_parameter("1", "xavier", [3, 3, n_channels, 32]),
                strides=[1, 2, 2, 1], padding="VALID")
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)

            net = tf.nn.conv2d(
                net,
                filter=ImageClassification.get_parameter("2", "xavier", [3, 3, 32, 32]),
                strides=[1, 1, 1, 1], padding="VALID")
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)

            net = tf.nn.conv2d(
                net,
                filter=ImageClassification.get_parameter("3", "xavier", [3, 3, 32, 64]),
                strides=[1, 1, 1, 1], padding="SAME")
            net = tf.layers.batch_normalization(net, training=is_training)
            net = tf.nn.relu(net)

            conv = tf.nn.conv2d(
                net,
                filter=ImageClassification.get_parameter("4", "xavier", [3, 3, 64, 96]),
                strides=[1, 2, 2, 1], padding="VALID")
            conv = tf.layers.batch_normalization(conv, training=is_training)
            conv = tf.nn.relu(conv)

            pool = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

            net = tf.concat([conv, pool], axis=-1)

            # Second part
            net1 = tf.nn.conv2d(
                net,
                filter=ImageClassification.get_parameter("5", "xavier", [1, 1, 160, 64]),
                strides=[1, 1, 1, 1], padding="SAME")
            net1 = tf.layers.batch_normalization(net1, training=is_training)
            net1 = tf.nn.relu(net1)

            net1 = tf.nn.conv2d(
                net1,
                filter=ImageClassification.get_parameter("6", "xavier", [7, 1, 64, 64]),
                strides=[1, 1, 1, 1], padding="SAME")
            net1 = tf.layers.batch_normalization(net1, training=is_training)
            net1 = tf.nn.relu(net1)

            net1 = tf.nn.conv2d(
                net1,
                filter=ImageClassification.get_parameter("7", "xavier", [1, 7, 64, 64]),
                strides=[1, 1, 1, 1], padding="SAME")
            net1 = tf.layers.batch_normalization(net1, training=is_training)
            net1 = tf.nn.relu(net1)

            net1 = tf.nn.conv2d(
                net1,
                filter=ImageClassification.get_parameter("8", "xavier", [3, 3, 64, 96]),
                strides=[1, 1, 1, 1], padding="VALID")
            net1 = tf.layers.batch_normalization(net1, training=is_training)
            net1 = tf.nn.relu(net1)

            net2 = tf.nn.conv2d(
                net,
                filter=ImageClassification.get_parameter("9", "xavier", [1, 1, 160, 64]),
                strides=[1, 1, 1, 1], padding="SAME")
            net2 = tf.layers.batch_normalization(net2, training=is_training)
            net2 = tf.nn.relu(net2)

            net2 = tf.nn.conv2d(
                net2,
                filter=ImageClassification.get_parameter("10", "xavier", [3, 3, 64, 96]),
                strides=[1, 1, 1, 1], padding="VALID")
            net2 = tf.layers.batch_normalization(net2, training=is_training)
            net2 = tf.nn.relu(net2)

            net = tf.concat([net1, net2], axis=-1)

            # Third part
            conv = tf.nn.conv2d(
                net,
                filter=ImageClassification.get_parameter("11", "xavier", [3, 3, 192, 192]),
                strides=[1, 2, 2, 1], padding="VALID")
            conv = tf.layers.batch_normalization(conv, training=is_training)
            conv = tf.nn.relu(conv)

            pool = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

            return tf.concat([conv, pool], axis=-1)

    def build_model(self, dim_output=1000, is_training=True, fc_padding='SAME', name=None):
        """
        Build the ResNet graph model.

        Args:
            dim_output: the dimension output of the network
            is_training: whether to use training layers
            fc_padding: the padding used for the fully connected layers
            name: the name of the graph operations

        Returns:
            the last layer of the model
        """
        with tf.variable_scope(name, 'InceptionV4', [self.input]) as sc:

            net = self.stem(self.input, n_channels=self.n_channel, is_training=is_training, scope="Stem")

            # 4 x inception A
            net = self.inception_A(net, n_channels=384, is_training=is_training, scope="InceptionA_1")
            net = self.inception_A(net, n_channels=384, is_training=is_training, scope="InceptionA_2")
            net = self.inception_A(net, n_channels=384, is_training=is_training, scope="InceptionA_3")
            net = self.inception_A(net, n_channels=384, is_training=is_training, scope="InceptionA_4")

            net = self.reduction_A(net, klmn=(192, 224, 256, 384), n_channels=384, is_training=is_training, scope="ReductionA")

            # 7 x inception B
            net = self.inception_B(net, n_channels=1024, is_training=is_training, scope="InceptionB_1")
            net = self.inception_B(net, n_channels=1024, is_training=is_training, scope="InceptionB_2")
            net = self.inception_B(net, n_channels=1024, is_training=is_training, scope="InceptionB_3")
            net = self.inception_B(net, n_channels=1024, is_training=is_training, scope="InceptionB_4")
            net = self.inception_B(net, n_channels=1024, is_training=is_training, scope="InceptionB_5")
            net = self.inception_B(net, n_channels=1024, is_training=is_training, scope="InceptionB_6")
            net = self.inception_B(net, n_channels=1024, is_training=is_training, scope="InceptionB_7")

            net = self.reduction_B(net, n_channels=1024, is_training=is_training, scope="ReductionB")

            # 3 x inception C
            net = self.inception_C(net, n_channels=1536, is_training=is_training, scope="InceptionC_1")
            net = self.inception_C(net, n_channels=1536, is_training=is_training, scope="InceptionC_2")
            net = self.inception_C(net, n_channels=1536, is_training=is_training, scope="InceptionC_3")

            net = tf.nn.avg_pool(net, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding="SAME")

            if self.n_classes:
                net = tf.nn.dropout(net, 0.8, name='dropout6')
                net = tf.nn.conv2d(net,
                       filter=ImageClassification.get_parameter("fc6", "xavier", [1, 1, 1536, self.n_classes]),
                       strides=[1, 1, 1, 1], padding=fc_padding, name='fc6')
                net = tf.squeeze(net, [1, 2], name='fc6/squeezed')
            return net
