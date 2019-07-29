import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from computer_vision.classImageClassification import ImageClassification

slim = tf.contrib.slim


class Vgg(ImageClassification):

    """
    The Vgg model is convolution network. The architecture is described in the following
    paper: https://arxiv.org/pdf/1409.1556.pdf

    """

    def __init__(self, classes, batch_size=1, height=224, width=224, dim_out=10,
                 grayscale=True, binarize=True, normalize=False,
                 learning_rate=10, n_epochs=1, validation_step=10,
                 checkpoint_step=100, is_encoder=True, validation_size=10,
                 optimizer="adam", metadata_path="", name="vgg",
                 from_pretrained=False, is_training=True, logger=None, debug=False):
        """
        Initialization of the Vgg model.

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
        self.model, self.end_points = self.build_model(self.dim_out, is_training=is_training)

        # Global step
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Optimizer
        self.optimizer = ImageClassification.get_optimizer(self.optimizer_name, self.learning_rate)

        # Summary writers
        self.train_writer, self.validation_writer = ImageClassification.get_writer(self)

        # Model saver
        self.saver = tf.train.Saver()

    def build_model(self, dim_output=1000, is_training=True, fc_padding='SAME', name=None):
        """
        Build the vgg graph model.

        Args:
            dim_output: the dimension output of the network
            is_training: whether to use training layers
            fc_padding: the padding used for the fully connected layers
            name: the name of the graph operations

        Returns:
            the last layer of the model
        """
        with tf.variable_scope(name, 'vgg_16', [self.input]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                    activation_fn=tf.nn.relu,
                                    padding="SAME",
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    biases_initializer=tf.zeros_initializer()):
                    net = slim.repeat(self.input, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')

                    # Use conv2d instead of fully_connected layers.
                    net = slim.conv2d(net, 4096, [7, 7], padding=fc_padding, scope='fc6')
                    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout6')
                    net = slim.conv2d(net, 4096, [1, 1], padding=fc_padding, scope='fc7')

                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                    if self.n_classes:
                        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout7')
                        net = slim.conv2d(net, self.n_classes, [1, 1],
                                          padding=fc_padding, activation_fn=None,
                                          normalizer_fn=None,
                                          scope='fc8')
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                        end_points[sc.name + '/fc8'] = net
                    return net, end_points

    def plot_features_maps(self, input, max_filters=64):
        """

        Args:
            input: the input image path
            max_filters: the maximum number of filters to plot

        Returns:
            Nothing
        """
        ixs = ['vgg_16/conv1/conv1_2',
               'vgg_16/conv2/conv2_2',
               'vgg_16/conv3/conv3_2',
               'vgg_16/conv4/conv4_2',
               'vgg_16/conv5/conv5_2']
        outputs = [self.end_points[i] for i in ixs]

        with tf.Session() as sess:

            # Load existing model
            ImageClassification.load(self, sess)

            image = self.load_example(input, with_labels=False)

            feature_maps = sess.run(
                outputs,
                feed_dict={
                    self.input: np.stack([image])
                }
            )

            # plot the output from each block
            square = int(np.sqrt(max_filters))
            for fmap in feature_maps:
                # plot all 64 maps in an 8x8 squares
                ix = 1
                for _ in range(square):
                    for _ in range(square):
                        # specify subplot and turn of axis
                        ax = plt.subplot(square, square, ix)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # plot filter channel in grayscale
                        plt.imshow(fmap[0, :, :, ix - 1], cmap='gray')
                        ix += 1
                # show the figure
                plt.show()