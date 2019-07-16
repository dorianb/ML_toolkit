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
        self.input = tf.placeholder(shape=(None, None, None, self.n_channel), dtype=tf.float32)

        # Label
        self.label = tf.placeholder(shape=(None, self.n_classes), dtype=tf.float32)
        self.label = tf.Print(self.label, [self.label], message="Truth label: ",
                              summarize=self.n_classes * self.batch_size) if self.debug else self.label

        # Build model
        self.model = self.build_model(self.dim_out)

        # Global step
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Optimizer
        self.optimizer = ComputerVision.get_optimizer(self.optimizer_name, self.learning_rate)

        # Summary writers
        self.train_writer, self.validation_writer = ComputerVision.get_writer(self)

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
                    return net

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
        loss = ComputerVision.compute_loss(self.model, self.label,
                                           loss_name="softmax_cross_entropy")

        # Compute probabilities
        logit = tf.nn.softmax(self.model)
        logit = tf.Print(logit, [logit], message="Probabilities: ",
                         summarize=self.n_classes * self.batch_size) if self.debug else logit

        # Accuracy
        accuracy = ComputerVision.compute_accuracy(self, logit, self.label)

        # Optimization
        train_op = ComputerVision.compute_gradient(self, loss, self.global_step)

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

            self.global_step = tf.add(self.global_step, tf.constant(1))

            for epoch in range(self.n_epochs):

                for i in range(self.batch_size, len(training_set)+self.batch_size, self.batch_size):

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
                        "Accuracy = {0}, Cost = {1} for batch {2} of epoch {3} in {4:.2f} seconds".format(
                            accuracy_value, loss_value, i / self.batch_size, epoch, time1 - time0)) if self.logger else None

                    if i % self.validation_step == 0:

                        self.validation_eval(sess, summaries,
                                             validation_set[:self.validation_size],
                                             step)

                    if i % self.checkpoint_step == 0:

                        ComputerVision.save(self, sess, step=self.global_step)

    def load_batch(self, examples, with_labels=True):
        """
        Load the batch examples.

        Args:
            examples: the example in the batch
            with_labels: whether to return label
        Returns:
            the batch examples
        """
        images = []
        labels = []
        for example in examples:
            if with_labels:
                image, label = self.load_example(example)
                images.append(image)
                labels.append(label)
            else:
                image = self.load_example(example, with_labels=with_labels)
                images.append(image)

        if with_labels:
            return np.stack(images), np.stack(labels)
        else:
            return np.stack(images)

    def load_example(self, example, with_labels=True):
        """
        Load the example.

        Args:
            example: an example with image path and label
            with_labels: whether to return label
        Returns:
            the example image array and label
        """
        if with_labels:
            image_path, label_id = example
            self.logger.info("Loading example: {0} with label {1}".format(
                image_path, label_id)) if self.logger else None

        else:
            image_path = example
            self.logger.info("Loading example: {0}".format(
                image_path)) if self.logger else None

        image = load_image(image_path, grayscale=self.grayscale,
                                          binarize=self.binarize,
                                          normalize=self.normalize,
                                          resize_dim=self.resize_dim)
        if with_labels:
            label = np.zeros(self.n_classes)
            label[int(label_id)] = 1

            return image, label
        else:
            return image

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

        # Compute probabilities
        logit = tf.nn.softmax(self.model)

        # Get predictions
        pred = tf.argmax(logit, axis=-1)

        with tf.Session() as sess:

            # Load existing model
            ComputerVision.load(self, sess)

            predictions = []

            for i in range(self.batch_size, len(dataset)+self.batch_size, self.batch_size):

                batch_examples = dataset[i - self.batch_size: i]

                image_batch = self.load_batch(batch_examples, with_labels=False)

                pred_batch = sess.run(
                    pred,
                    feed_dict={
                        self.input: image_batch
                    }
                )

                predictions += pred_batch.tolist()

            return predictions