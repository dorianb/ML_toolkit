import os
import tensorflow as tf
import numpy as np
from time import time

from computer_vision.classComputerVision import ComputerVision
from dataset_utils.functionImageUtils import load_image

slim = tf.contrib.slim


class Vgg(ComputerVision):

    """
    The Vgg model is convolution network. The architecture is described in the following
    paper: https://arxiv.org/pdf/1409.1556.pdf

    """

    def __init__(self, classes, batch_size=1, height=224, width=224, dim_out=10,
                 grayscale=True, binarize=True, normalize=False,
                 learning_rate=10, n_epochs=1, validation_step=10,
                 checkpoint_step=100, is_encoder=True, validation_size=10,
                 optimizer="adam", metadata_path="", name="vgg",
                 from_pretrained=False, logger=None, debug=False):
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
            logger: an instance object of logging module.
            debug: whether the debug mode is activated or not

        Returns:
            Nothing
        """
        tf.reset_default_graph()

        ComputerVision.__init__(self)

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
        self.input = tf.placeholder(
            shape=(None, None, None, self.n_channel),
            dtype=tf.float32)

        # Label
        self.label = tf.placeholder(shape=(None, self.n_classes),
                                    dtype=tf.float32)
        self.label = tf.Print(self.label, [self.label], message="Truth label: ",
                              summarize=self.n_classes * self.batch_size) if self.debug else self.label

        # Build model
        self.model = self.build_model(self.dim_out)

        # Global step
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Optimizer
        self.optimizer = ComputerVision.get_optimizer(self.optimizer_name,
                                                      self.learning_rate)

        # Summary writers
        self.train_writer, self.validation_writer = ComputerVision.get_writer(self)

        # Model saver
        self.saver = tf.train.Saver()

    @staticmethod
    def initialize_variable(name, shape):
        """
        Initialize a variable

        Args:
            name: the name of the variable
            shape: the shape of the variable

        Returns:
            tensorflow variable
        """
        # initializer = tf.random_uniform_initializer(minval=0, maxval=0.0001, seed=42)
        initializer = tf.zeros_initializer()
        # initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=42)
        variable = tf.get_variable(name, shape=shape, dtype=tf.float32,
                                   initializer=initializer)
        Vgg.variable_summaries(variable, name)
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
    def conv_layer(input, filter, bias, strides=[1, 1, 1, 1], padding='SAME',
                   activation=tf.nn.relu, name="conv"):
        """
        Convolution layer graph definition.

        Args:
            input: the input of convolution
            filter: the filter tensor
            bias: the bias tensor
            strides: the strides of convolution
            padding: the padding choice between SAME and VALID
            activation: activation function applied in output of the layer
            name: the name scope for the operations applied
        Returns:
            a tensor operation
        """
        with tf.name_scope(name, [input]):
            return activation(tf.nn.bias_add(
                tf.nn.conv2d(input, filter=filter, strides=strides, padding=padding),
                bias
            ))

    def build_model(self, dim_output=1000, fc_padding='VALID', name=None):
        """
        Build the vgg graph model.

        Args:
            dim_output: the dimension output of the network
            fc_padding: the padding used for the fully connected layers
            name: the name of the graph operations

        Returns:
            the last layer of the model
        """
        """with tf.name_scope(name, "VggOp", [self.input]):

            with tf.variable_scope("Parameters", reuse=tf.AUTO_REUSE):

                input = tf.Print(self.input, [tf.shape(self.input)], message="Input shape: ",
                                 summarize=4) if self.debug else self.input
                input = tf.Print(input, [input], message="Input: ",
                                 summarize=100) if self.debug else input

                # 2 x conv2D
                filter1_1 = Vgg.initialize_variable(
                    "filter1_1", shape=[3, 3, self.n_channel, 64])
                bias1_1 = Vgg.initialize_variable("bias1_1", shape=[64])

                conv1_1 = Vgg.conv_layer(input, filter=filter1_1, bias=bias1_1,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv1_1")

                conv1_1 = tf.Print(conv1_1, [tf.shape(conv1_1)], message="Conv1_1 shape:",
                                   summarize=4) if self.debug else conv1_1

                filter1_2 = Vgg.initialize_variable(
                    "filter1_2", shape=[3, 3, 64, 64])
                bias1_2 = Vgg.initialize_variable("bias1_2", shape=[64])

                conv1_2 = Vgg.conv_layer(conv1_1, filter=filter1_2, bias=bias1_2,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv1_2")

                conv1_2 = tf.Print(conv1_2, [tf.shape(conv1_2)], message="Conv1_2 shape:",
                                   summarize=4) if self.debug else conv1_2

                # Max pooling2D
                pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding='SAME', name="pool1")
                pool1 = tf.Print(pool1, [tf.shape(pool1)], message="Pool 1 shape:",
                                 summarize=4) if self.debug else pool1

                # 2 x conv2D
                filter2_1 = Vgg.initialize_variable(
                    "filter2_1", shape=[3, 3, 64, 128])
                bias2_1 = Vgg.initialize_variable("bias2_1", shape=[128])

                conv2_1 = Vgg.conv_layer(pool1, filter=filter2_1, bias=bias2_1,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv2_1")

                conv2_1 = tf.Print(conv2_1, [tf.shape(conv2_1)], message="Conv2_1 shape:",
                                   summarize=4) if self.debug else conv2_1

                filter2_2 = Vgg.initialize_variable(
                    "filter2_2", shape=[3, 3, 128, 128])
                bias2_2 = Vgg.initialize_variable("bias2_2", shape=[128])

                conv2_2 = Vgg.conv_layer(conv2_1, filter=filter2_2, bias=bias2_2,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv2_2")

                conv2_2 = tf.Print(conv2_2, [tf.shape(conv2_2)], message="Conv2_2 shape:",
                                   summarize=4) if self.debug else conv2_2

                # Max pooling2D
                pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name="pool2")
                pool2 = tf.Print(pool2, [tf.shape(pool2)], message="Pool 2 shape:",
                                 summarize=4) if self.debug else pool2

                # 3 x conv2D
                filter3_1 = Vgg.initialize_variable(
                    "filter3_1", shape=[3, 3, 128, 256])
                bias3_1 = Vgg.initialize_variable("bias3_1", shape=[256])

                conv3_1 = Vgg.conv_layer(pool2, filter=filter3_1, bias=bias3_1,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv3_1")

                conv3_1 = tf.Print(conv3_1, [tf.shape(conv3_1)], message="conv3_1 shape:",
                                   summarize=4) if self.debug else conv3_1

                filter3_2 = Vgg.initialize_variable(
                    "filter3_2", shape=[3, 3, 256, 256])
                bias3_2 = Vgg.initialize_variable("bias3_2", shape=[256])

                conv3_2 = Vgg.conv_layer(conv3_1, filter=filter3_2, bias=bias3_2,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv3_2")

                conv3_2 = tf.Print(conv3_2, [tf.shape(conv3_2)], message="conv3_2 shape:",
                                   summarize=4) if self.debug else conv3_2

                filter3_3 = Vgg.initialize_variable(
                    "filter3_3", shape=[3, 3, 256, 256])
                bias3_3 = Vgg.initialize_variable("bias3_3", shape=[256])

                conv3_3 = Vgg.conv_layer(conv3_2, filter=filter3_3, bias=bias3_3,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv3_3")

                conv3_3 = tf.Print(conv3_3, [tf.shape(conv3_3)], message="Conv3_3 shape:",
                                   summarize=4) if self.debug else conv3_3

                # Max pooling2D
                pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding='SAME', name="pool3")
                pool3 = tf.Print(pool3, [tf.shape(pool3)], message="Pool 3 shape:",
                                 summarize=4) if self.debug else pool3

                # 3 x conv2D
                filter4_1 = Vgg.initialize_variable(
                    "filter4_1", shape=[3, 3, 256, 512])
                bias4_1 = Vgg.initialize_variable("bias4_1", shape=[512])

                conv4_1 = Vgg.conv_layer(pool3, filter=filter4_1, bias=bias4_1,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv4_1")

                conv4_1 = tf.Print(conv4_1, [tf.shape(conv4_1)], message="Conv4_1 shape:",
                                   summarize=4) if self.debug else conv4_1

                filter4_2 = Vgg.initialize_variable(
                    "filter4_2", shape=[3, 3, 512, 512])
                bias4_2 = Vgg.initialize_variable("bias4_2", shape=[512])

                conv4_2 = Vgg.conv_layer(conv4_1, filter=filter4_2, bias=bias4_2,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv4_2")

                conv4_2 = tf.Print(conv4_2, [tf.shape(conv4_2)], message="Conv4_2 shape:",
                                   summarize=4) if self.debug else conv4_2

                filter4_3 = Vgg.initialize_variable(
                    "filter4_3", shape=[3, 3, 512, 512])
                bias4_3 = Vgg.initialize_variable("bias4_3", shape=[512])

                conv4_3 = Vgg.conv_layer(conv4_2, filter=filter4_3, bias=bias4_3,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv4_3")

                conv4_3 = tf.Print(conv4_3, [tf.shape(conv4_3)], message="Conv4_3 shape:",
                                   summarize=4) if self.debug else conv4_3

                # Max pooling2D
                pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name="pool4")
                pool4 = tf.Print(pool4, [tf.shape(pool4)], message="Pool 4 shape:",
                                 summarize=4) if self.debug else pool4

                # 3 x conv2D
                filter5_1 = Vgg.initialize_variable(
                    "filter5_1", shape=[3, 3, 512, 512])
                bias5_1 = Vgg.initialize_variable("bias5_1", shape=[512])

                conv5_1 = Vgg.conv_layer(pool4, filter=filter5_1, bias=bias5_1,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv5_1")

                conv5_1 = tf.Print(conv5_1, [tf.shape(conv5_1)], message="Conv5_1 shape:",
                                   summarize=4) if self.debug else conv5_1

                filter5_2 = Vgg.initialize_variable(
                    "filter5_2", shape=[3, 3, 512, 512])
                bias5_2 = Vgg.initialize_variable("bias5_2", shape=[512])

                conv5_2 = Vgg.conv_layer(conv5_1, filter=filter5_2, bias=bias5_2,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv5_2")

                conv5_2 = tf.Print(conv5_2, [tf.shape(conv5_2)], message="Conv5_2 shape:",
                                   summarize=4) if self.debug else conv5_2

                filter5_3 = Vgg.initialize_variable(
                    "filter5_3", shape=[3, 3, 512, 512])
                bias5_3 = Vgg.initialize_variable("bias5_3", shape=[512])

                conv5_3 = Vgg.conv_layer(conv5_2, filter=filter5_3, bias=bias5_3,
                                         strides=[1, 1, 1, 1], padding='SAME',
                                         activation=tf.nn.relu, name="conv5_3")

                conv5_3 = tf.Print(conv5_3, [tf.shape(conv5_3)], message="Conv5_3 shape:",
                                   summarize=4) if self.debug else conv5_3

                # Max pooling2D
                pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                       padding='SAME', name="pool5")
                pool5 = tf.Print(pool5, [tf.shape(pool5)], message="Pool 5 shape:",
                                 summarize=4) if self.debug else pool5

                # 3 x Dense
                filter6 = Vgg.initialize_variable(
                    "filter6", shape=[7, 7, 512, 4096])
                bias6 = Vgg.initialize_variable("bias6", shape=[4096])

                fc6 = Vgg.conv_layer(pool5, filter=filter6, bias=bias6,
                                     strides=[1, 1, 1, 1], padding=fc_padding,
                                     activation=tf.nn.relu, name="fc6")

                fc6 = tf.Print(fc6, [tf.shape(fc6)], message="fc6 shape:",
                                summarize=4) if self.debug else fc6

                dropout6 = tf.nn.dropout(fc6, keep_prob=0.5, name="dropout6")

                if self.is_encoder:

                    return dropout6

                else:

                    filter7 = Vgg.initialize_variable(
                        "filter7", shape=[1, 1, 4096, 4096])
                    bias7 = Vgg.initialize_variable("bias7", shape=[4096])

                    fc7 = Vgg.conv_layer(dropout6, filter=filter7, bias=bias7,
                                         strides=[1, 1, 1, 1], padding=fc_padding,
                                         activation=tf.nn.relu, name="fc7")

                    fc7 = tf.Print(fc7, [tf.shape(fc7)], message="fc7 shape:",
                                   summarize=4) if self.debug else fc7

                    dropout7 = tf.nn.dropout(fc7, keep_prob=0.5, name="dropout7")

                    filter8 = Vgg.initialize_variable(
                        "filter8", shape=[1, 1, 4096, dim_output])
                    bias8 = Vgg.initialize_variable("bias8", shape=[dim_output])

                    fc8 = Vgg.conv_layer(dropout7, filter=filter8, bias=bias8,
                                         strides=[1, 1, 1, 1], padding=fc_padding,
                                         activation=tf.identity, name="fc8")

                    fc8 = tf.Print(fc8, [tf.shape(fc8)], message="fc8 shape:",
                                   summarize=4) if self.debug else fc8

                    output = tf.squeeze(fc8)
                    output = tf.Print(output, [output], message="Last layer: ",
                                      summarize=self.n_classes * self.batch_size) if self.debug else output

                    return output
        """
        with tf.variable_scope(name, 'vgg_16', [self.input]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
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
                net = slim.dropout(net, 0.5, is_training=True,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                if self.n_classes:
                    net = slim.dropout(net, 0.5, is_training=True,
                                       scope='dropout7')
                    net = slim.conv2d(net, self.n_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')

                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net

    def compute_loss(self, logit, label):
        """
        Compute the loss operation.

        Args:
            logit: the tensor of class probabilities (bach_size, n_classes)
            label: the tensor of labels (batch_size, n_classes)

        Returns:
            loss: the loss
        """
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logit, labels=label
        ))
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        #    logits=logit, labels=label
        #))
        tf.summary.scalar('Cross_entropy', loss)
        return loss

    def compute_accuracy(self, logit, label):
        """
        Compute the accuracy measure.

        Args:
            logit: the tensor of class probabilities (bach_size, n_classes)
            label: the tensor of labels (batch_size, n_classes)

        Returns:
            accuracy: the accuracy metric measure

        """
        pred = tf.argmax(logit, axis=-1)
        y = tf.argmax(label, axis=-1)

        pred = tf.Print(pred, [pred], message="Prediction: ",
                        summarize=2) if self.debug else pred
        y = tf.Print(y, [y], message="Label: ",
                     summarize=2) if self.debug else y

        accuracy = tf.reduce_mean(tf.cast(tf.equal(y, pred), "float"))
        tf.summary.scalar('Accuracy', accuracy)

        return accuracy

    def compute_gradient(self, loss, global_step):
        """
        Compute gradient and update parameters.

        Args:
            loss: the loss to minimize
            global_step: the training step
        Returns:
            the training operation
        """
        grads_and_vars = self.optimizer.compute_gradients(loss)
        self.logger.debug(grads_and_vars) if self.logger else None
        return self.optimizer.apply_gradients(grads_and_vars,
                                              global_step=global_step)

    def fit(self, training_set, validation_set):
        """
        Fit the model weights with input and labels.

        Args:
            training_set: the training input and label
            validation_set: the validation input and label

        Returns:
            Nothing
        """

        if self.is_encoder:
            raise Exception("Vgg Fit method is implemented for image classification "
                            "purpose only")

        # Loss
        loss = self.compute_loss(self.model, self.label)

        # Compute probabilities
        logit = tf.nn.softmax(self.model)
        logit = tf.Print(logit, [logit], message="Probabilities: ",
                         summarize=self.n_classes * self.batch_size) if self.debug else logit

        # Accuracy
        accuracy = self.compute_accuracy(logit, self.label)

        # Optimization
        train_op = self.compute_gradient(loss, self.global_step)

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

                    self.global_step = tf.add(self.global_step, tf.constant(1))

                    time0 = time()
                    batch_examples = training_set[i - self.batch_size: i]

                    image_batch, label_batch = self.load_batch(batch_examples)

                    _, loss_value, summaries_value, accuracy_value, step = sess.run([
                        train_op, loss, summaries, accuracy, self.global_step],
                        feed_dict={
                            self.input: image_batch,
                            self.label: label_batch
                        }
                    )

                    self.logger.info("Writing summary to {0}".format(self.summary_path)) if self.logger else None
                    self.train_writer.add_summary(summaries_value, step)

                    time1 = time()
                    self.logger.info(
                        "Accuracy = {0}, Cost = {1} for batch {2} in {3:.2f} seconds".format(
                            accuracy_value, loss_value, i / self.batch_size, time1 - time0)) if self.logger else None

                    if i % self.validation_step == 0:

                        self.validation_eval(sess, summaries,
                                             validation_set[:self.validation_size],
                                             step)

                    if i % self.checkpoint_step == 0:

                        ComputerVision.save(self, sess, step=self.global_step)

    def load_batch(self, examples):
        """
        Load the batch examples.

        Args:
            examples: the example in the batch

        Returns:
            the batch examples
        """
        images = []
        labels = []
        for example in examples:
            image, label = self.load_example(example)
            images.append(image)
            labels.append(label)

        return np.stack(images), np.stack(labels)

    def load_example(self, example):
        """
        Load the example.

        Args:
            example: an example with image path and label

        Returns:
            the example image array and label
        """
        image_path, label_id = example
        self.logger.info("Loading example: {0} with label {1}".format(
            image_path, label_id)) if self.logger else None

        image = load_image(image_path, grayscale=self.grayscale,
                                          binarize=self.binarize,
                                          normalize=self.normalize,
                                          resize_dim=self.resize_dim)

        label = np.zeros(self.n_classes)
        label[int(label_id)] = 1

        return image, label

    def validation_eval(self, session, summaries, dataset, step):
        """
        Produce evaluation on the validation dataset.

        Args:
            session: the session object opened
            summaries: the summaries declared in the graph
            dataset: the dataset to use for validation
            step: the step of summarize writing

        Returns:
            Nothing
        """
        images, labels = self.load_batch(dataset)

        summaries_value = session.run(
            summaries,
            feed_dict={
                self.input: images,
                self.label: labels
            }
        )

        self.validation_writer.add_summary(summaries_value, step)

    def predict(self, dataset):
        """
        Predict the output from input.

        Args:
            dataset: the input dataset

        Returns:
            predictions array
        """
        pass