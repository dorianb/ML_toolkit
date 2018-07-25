import tensorflow as tf
import numpy as np
import os
from time import time
from computer_vision.classComputerVision import ComputerVision


class Vgg(ComputerVision):

    """
    The Vgg model is convolution network. The architecture is described in the following
    paper: https://arxiv.org/pdf/1409.1556.pdf

    """

    def __init__(self, classes, batch_size=1, height=224, width=224, dim_out=10,
                 grayscale=True, binarize=True, normalize=False,
                 learning_rate=10, n_epochs=1, validation_step=10,
                 is_encoder=True, validation_size=10, summary_path="",
                 logger=None, debug=False):
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
            is_encoder: the vgg is used as an encoder
            validation_size: the number of examples to use for validation
            summary_path: the path to the summary
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
        self.is_encoder = is_encoder
        self.validation_size = validation_size
        self.summary_path = summary_path
        self.logger = logger
        self.debug = debug

        # Input
        self.input = tf.placeholder(
            shape=(self.batch_size, None, None, self.n_channel),
            dtype=tf.float32)

        # Label
        self.label = tf.placeholder(shape=(self.batch_size, self.n_classes),
                                    dtype=tf.float32)

        # Build model
        self.model = self.build_model(self.dim_out)

        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

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

    def build_model(self, dim_output=1000, name=None):
        """
        Build the vgg graph model.

        Args:
            dim_output: the dimension output of the network
            name: the name of the graph operations

        Returns:
            the last layer of the model
        """
        with tf.name_scope(name, "VggOp", [self.input]):

            with tf.variable_scope("Parameters", reuse=tf.AUTO_REUSE):

                # 2 x conv2D
                conv1_1 = tf.nn.relu(tf.nn.conv2d(self.input, filter=Vgg.initialize_variable(
                                                    "filter1_1", shape=[3, 3, self.n_channel, 64]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv1_1"))

                conv1_1 = tf.Print(conv1_1, [tf.shape(conv1_1)], message="Conv1_1 shape:",
                                   summarize=4) if self.debug else conv1_1

                conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1, filter=Vgg.initialize_variable(
                                                    "filter1_2", shape=[3, 3, 64, 64]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv1_2"))
                conv1_2 = tf.Print(conv1_2, [tf.shape(conv1_2)], message="Conv1_2 shape:",
                                   summarize=4) if self.debug else conv1_2

                # Max pooling2D
                pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding='SAME', name="pool1")
                pool1 = tf.Print(pool1, [tf.shape(pool1)], message="Pool 1 shape:",
                                 summarize=4) if self.debug else pool1

                # 2 x conv2D
                conv2_1 = tf.nn.relu(tf.nn.conv2d(pool1, filter=Vgg.initialize_variable(
                                                    "filter2_1", shape=[3, 3, 64, 128]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv2_1"))
                conv2_1 = tf.Print(conv2_1, [tf.shape(conv2_1)], message="Conv2_1 shape:",
                                   summarize=4) if self.debug else conv2_1

                conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1, filter=Vgg.initialize_variable(
                                                    "filter2_2", shape=[3, 3, 128, 128]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv2_2"))
                conv2_2 = tf.Print(conv2_2, [tf.shape(conv2_2)], message="Conv2_2 shape:",
                                   summarize=4) if self.debug else conv2_2

                # Max pooling2D
                pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding='SAME', name="pool2")
                pool2 = tf.Print(pool2, [tf.shape(pool2)], message="Pool 2 shape:",
                                 summarize=4) if self.debug else pool2

                # 3 x conv2D
                conv3_1 = tf.nn.relu(tf.nn.conv2d(pool2, filter=Vgg.initialize_variable(
                                                    "filter3_1", shape=[3, 3, 128, 256]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv3_1"))
                conv3_1 = tf.Print(conv3_1, [tf.shape(conv3_1)], message="conv3_1 shape:",
                                   summarize=4) if self.debug else conv3_1

                conv3_2 = tf.nn.relu(tf.nn.conv2d(conv3_1, filter=Vgg.initialize_variable(
                                                    "filter3_2", shape=[3, 3, 256, 256]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv3_2"))
                conv3_2 = tf.Print(conv3_2, [tf.shape(conv3_2)], message="conv3_2 shape:",
                                   summarize=4) if self.debug else conv3_2

                conv3_3 = tf.nn.relu(tf.nn.conv2d(conv3_2, filter=Vgg.initialize_variable(
                                                    "filter3_3", shape=[3, 3, 256, 256]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv3_3"))
                conv3_3 = tf.Print(conv3_3, [tf.shape(conv3_3)], message="Conv3_3 shape:",
                                   summarize=4) if self.debug else conv3_3

                # Max pooling2D
                pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                        padding='SAME', name="pool3")
                pool3 = tf.Print(pool3, [tf.shape(pool3)], message="Pool 3 shape:",
                                 summarize=4) if self.debug else pool3

                # 3 x conv2D
                conv4_1 = tf.nn.relu(tf.nn.conv2d(pool3, filter=Vgg.initialize_variable(
                                                    "filter4_1", shape=[3, 3, 256, 512]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv4_1"))
                conv4_1 = tf.Print(conv4_1, [tf.shape(conv4_1)], message="Conv4_1 shape:",
                                   summarize=4) if self.debug else conv4_1

                conv4_2 = tf.nn.relu(tf.nn.conv2d(conv4_1, filter=Vgg.initialize_variable(
                                                    "filter4_2", shape=[3, 3, 512, 512]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv4_2"))
                conv4_2 = tf.Print(conv4_2, [tf.shape(conv4_2)], message="Conv4_2 shape:",
                                   summarize=4) if self.debug else conv4_2

                conv4_3 = tf.nn.relu(tf.nn.conv2d(conv4_2, filter=Vgg.initialize_variable(
                                                    "filter4_3", shape=[3, 3, 512, 512]),
                                     strides=[1, 1, 1, 1], padding='SAME', name="conv4_3"))
                conv4_3 = tf.Print(conv4_3, [tf.shape(conv4_3)], message="Conv4_3 shape:",
                                   summarize=4) if self.debug else conv4_3

                # Max pooling2D
                pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name="pool4")
                pool4 = tf.Print(pool4, [tf.shape(pool4)], message="Pool 4 shape:",
                                 summarize=4) if self.debug else pool4

                # 3 x conv2D
                conv5_1 = tf.nn.relu(tf.nn.conv2d(pool4, filter=Vgg.initialize_variable(
                                                    "filter5_1", shape=[3, 3, 512, 512]),
                                      strides=[1, 1, 1, 1], padding='SAME', name="conv5_1"))
                conv5_1 = tf.Print(conv5_1, [tf.shape(conv5_1)], message="Conv5_1 shape:",
                                   summarize=4) if self.debug else conv5_1

                conv5_2 = tf.nn.relu(tf.nn.conv2d(conv5_1, filter=Vgg.initialize_variable(
                                                    "filter5_2", shape=[3, 3, 512, 512]),
                                      strides=[1, 1, 1, 1], padding='SAME', name="conv5_2"))
                conv5_2 = tf.Print(conv5_2, [tf.shape(conv5_2)], message="Conv5_2 shape:",
                                   summarize=4) if self.debug else conv5_2

                conv5_3 = tf.nn.relu(tf.nn.conv2d(conv5_2, filter=Vgg.initialize_variable(
                                                    "filter5_3", shape=[3, 3, 512, 512]),
                                      strides=[1, 1, 1, 1], padding='SAME', name="conv5_3"))
                conv5_3 = tf.Print(conv5_3, [tf.shape(conv5_3)], message="Conv5_3 shape:",
                                   summarize=4) if self.debug else conv5_3

                # Max pooling2D
                pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name="pool5")
                pool5 = tf.Print(pool5, [tf.shape(pool5)], message="Pool 5 shape:",
                                 summarize=4) if self.debug else pool5

                # 3 x Dense
                fc6 = tf.nn.relu(tf.nn.conv2d(pool5, filter=Vgg.initialize_variable(
                                                    "filter6", shape=[7, 7, 512, 4096]),
                                      strides=[1, 7, 7, 1], padding='SAME', name="fc6"))
                fc6 = tf.Print(fc6, [tf.shape(fc6)], message="fc6 shape:",
                               summarize=4) if self.debug else fc6

                if self.is_encoder:

                    return fc6

                else:

                    fc7 = tf.nn.relu(tf.nn.conv2d(fc6, filter=Vgg.initialize_variable(
                                                    "filter7", shape=[1, 1, 4096, 4096]),
                                        strides=[1, 1, 1, 1], padding='SAME', name="fc7"))
                    fc7 = tf.Print(fc7, [tf.shape(fc7)], message="fc7 shape:",
                                   summarize=4) if self.debug else fc7

                    fc8 = tf.nn.relu(tf.nn.conv2d(fc7, filter=Vgg.initialize_variable(
                                                    "filter8", shape=[1, 1, 4096, dim_output]),
                                        strides=[1, 1, 1, 1], padding='SAME', name="fc8"))
                    fc8 = tf.Print(fc8, [tf.shape(fc8)], message="fc8 shape:",
                                   summarize=4) if self.debug else fc8

                    return fc8

    def compute_loss(self, logit, label):
        """
        Compute the loss operation.

        Args:
            logit: the tensor of class probabilities (bach_size, n_classes)
            label: the tensor of labels (batch_size, n_classes)

        Returns:
            loss: the loss
        """
        """                                                               
        prod = tf.multiply(label, tf.log(logit))                          
        prod = tf.Print(prod, [prod], message="Prod: ",                   
                        summarize=self.n_classes * self.batch_size)       
        loss = tf.reduce_mean(-tf.reduce_sum(prod, axis=-1))              
        """
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                              logits=logit, labels=label))
        # loss = tf.Print(loss, [loss], message='Loss:')
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
        # accuracy = tf.metrics.accuracy(labels=y, predictions=pred)
        tf.summary.scalar('Accuracy', accuracy)

        return accuracy

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

        global_step = tf.Variable(0, dtype=tf.int32)

        # Compute probabilities
        model = tf.squeeze(self.model)
        model = tf.Print(model, [model], message="Last layer: ",
                         summarize=self.n_classes * self.batch_size) if self.debug else model

        logit = tf.nn.softmax(model)
        logit = tf.Print(logit, [logit], message="Probabilities: ",
                         summarize=self.n_classes * self.batch_size) if self.debug else logit
        label = tf.Print(self.label, [self.label], message="Truth label: ",
                         summarize=self.n_classes * self.batch_size) if self.debug else self.label

        # Loss
        loss = self.compute_loss(logit, label)

        # Accuracy
        accuracy = self.compute_accuracy(logit, label)

        # Minimization
        grads_and_vars = self.optimizer.compute_gradients(loss)
        self.logger.debug(grads_and_vars) if self.logger else None
        train_op = self.optimizer.apply_gradients(grads_and_vars,
                                                  global_step=global_step)

        # Merge summaries
        summaries = tf.summary.merge_all()

        # Initialize variables
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        with tf.Session() as sess:

            sess.run(init_g)
            sess.run(init_l)

            train_writer = tf.summary.FileWriter(
                os.path.join(self.summary_path, 'train'), sess.graph)
            validation_writer = tf.summary.FileWriter(
                os.path.join(self.summary_path, 'validation'))

            for epoch in range(self.n_epochs):

                for i in range(self.batch_size, len(training_set), self.batch_size):

                    time0 = time()
                    batch_examples = training_set[i - self.batch_size: i]

                    image_batch, label_batch = self.load_batch(batch_examples)

                    _, loss_value, summaries_value, accuracy_value = sess.run([
                        train_op, loss, summaries, accuracy],
                        feed_dict={
                            self.input: image_batch,
                            self.label: label_batch
                        }
                    )

                    self.logger.info("Writing summary to {0}".format(self.summary_path))
                    train_writer.add_summary(summaries_value, i * (epoch + 1))

                    time1 = time()
                    self.logger.info(
                        "Accuracy = {0}, Cost = {1} for batch {2} in {3:.2f} seconds".format(
                            accuracy_value, loss_value, i / self.batch_size, time1 - time0)) if self.logger else None

                    if i % self.validation_step == 0:

                        self.validation_eval()

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
        self.logger.info("Loading example: {0} with label {1}".format(image_path, label_id))

        image = ComputerVision.load_image(image_path, grayscale=self.grayscale,
                                          binarize=self.binarize,
                                          normalize=self.normalize,
                                          resize_dim=self.resize_dim)

        label = np.zeros(self.n_classes)
        label[int(label_id)] = 1

        return image, label

    def validation_eval(self):
        pass

    def predict(self, set):
        """
        Predict the output from input.

        Args:
            set: the input set

        Returns:
            predictions array
        """
        pass
