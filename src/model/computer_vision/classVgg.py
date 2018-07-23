import tensorflow as tf
import numpy as np
from computer_vision.classComputerVision import ComputerVision


class Vgg(ComputerVision):

    """
    The Vgg model is convolution network. The architecture is described in the following
    paper: https://arxiv.org/pdf/1409.1556.pdf

    """

    def __init__(self, batch_size=1, height=1200, width=800, dim_out=10,
                 grayscale=True, binarize=True, normalize=False,
                 n_classes=10, learning_rate=10, n_epochs=1, validation_step=10,
                 is_encoder=True, validation_size=10, logger=None):
        """
        Initialization of the Vgg model.

        Args:
            batch_size: the size of batch
            height: the height of the image
            width: the width of the image
            grayscale: whether the input image are grayscale
            binarize: whether input image are binarized
            normalize: whether input image are normalized
            dim_out: the output dimension of the model
            n_classes: the number of classes
            learning_rate: the learning rate applied in the gradient descent optimization
            n_epochs: the number of epochs
            validation_step: the number of training examples to use for training before
             evaluation on validation dataset
            is_encoder: the vgg is used as an encoder
            validation_size: the number of examples to use for validation
            logger: an instance object of logging module.

        Returns:
            Nothing
        """
        tf.reset_default_graph()

        ComputerVision.__init__(self)

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.grayscale = grayscale
        self.binarize = binarize
        self.normalize = normalize
        self.n_channel = 1 if self.grayscale else 3
        self.dim_out = dim_out
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.validation_step = validation_step
        self.is_encoder = is_encoder
        self.validation_size = validation_size
        self.logger = logger

        # Input
        self.input = tf.placeholder(
            # shape=(self.batch_size, 224, 224, 3),
            shape=(self.batch_size, None, None, self.n_channel),
            dtype=tf.float32)

        # Label
        self.label = tf.placeholder(shape=(self.batch_size, self.n_classes),
                                    dtype=tf.float32)

        # Build model
        self.model = self.build_model(self.dim_out)

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
        initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=42)
        return tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer)

    def build_model(self, dim_output=1000):
        """
        Build the vgg graph model.

        Args:
            dim_output: the dimension output of the network

        Returns:
            the last layer of the model
        """

        # 2 x conv2D
        conv1_1 = tf.nn.relu(tf.nn.conv2d(self.input, filter=Vgg.initialize_variable(
                                            "filter1_1", shape=[3, 3, self.n_channel, 64]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv1_1"))

        conv1_1 = tf.Print(conv1_1, [tf.shape(conv1_1)], message="Conv1_1 shape:")

        conv1_2 = tf.nn.relu(tf.nn.conv2d(conv1_1, filter=Vgg.initialize_variable(
                                            "filter1_2", shape=[3, 3, 64, 64]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv1_2"))
        conv1_2 = tf.Print(conv1_2, [tf.shape(conv1_2)], message="Conv1_2 shape:")

        # Max pooling2D
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name="pool1")
        pool1 = tf.Print(pool1, [tf.shape(pool1)], message="Pool 1 shape:")

        # 2 x conv2D
        conv2_1 = tf.nn.relu(tf.nn.conv2d(pool1, filter=Vgg.initialize_variable(
                                            "filter2_1", shape=[3, 3, 64, 128]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv2_1"))
        conv2_1 = tf.Print(conv2_1, [tf.shape(conv2_1)], message="Conv2_1 shape:")

        conv2_2 = tf.nn.relu(tf.nn.conv2d(conv2_1, filter=Vgg.initialize_variable(
                                            "filter2_2", shape=[3, 3, 128, 128]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv2_2"))
        conv2_2 = tf.Print(conv2_2, [tf.shape(conv2_2)], message="Conv2_2 shape:")

        # Max pooling2D
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name="pool2")
        pool2 = tf.Print(pool2, [tf.shape(pool2)], message="Pool 2 shape:")

        # 3 x conv2D
        conv3_1 = tf.nn.relu(tf.nn.conv2d(pool2, filter=Vgg.initialize_variable(
                                            "filter3_1", shape=[3, 3, 128, 256]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv3_1"))
        conv3_1 = tf.Print(conv3_1, [tf.shape(conv3_1)], message="conv3_1 shape:")

        conv3_2 = tf.nn.relu(tf.nn.conv2d(conv3_1, filter=Vgg.initialize_variable(
                                            "filter3_2", shape=[3, 3, 256, 256]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv3_2"))
        conv3_2 = tf.Print(conv3_2, [tf.shape(conv3_2)], message="conv3_2 shape:")

        conv3_3 = tf.nn.relu(tf.nn.conv2d(conv3_2, filter=Vgg.initialize_variable(
                                            "filter3_3", shape=[3, 3, 256, 256]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv3_3"))
        conv3_3 = tf.Print(conv3_3, [tf.shape(conv3_3)], message="Conv3_3 shape:")

        # Max pooling2D
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                padding='SAME', name="pool3")
        pool3 = tf.Print(pool3, [tf.shape(pool3)], message="Pool 3 shape:")

        # 3 x conv2D
        conv4_1 = tf.nn.relu(tf.nn.conv2d(pool3, filter=Vgg.initialize_variable(
                                            "filter4_1", shape=[3, 3, 256, 512]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv4_1"))
        conv4_1 = tf.Print(conv4_1, [tf.shape(conv4_1)], message="Conv4_1 shape:")

        conv4_2 = tf.nn.relu(tf.nn.conv2d(conv4_1, filter=Vgg.initialize_variable(
                                            "filter4_2", shape=[3, 3, 512, 512]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv4_2"))
        conv4_2 = tf.Print(conv4_2, [tf.shape(conv4_2)], message="Conv4_2 shape:")

        conv4_3 = tf.nn.relu(tf.nn.conv2d(conv4_2, filter=Vgg.initialize_variable(
                                            "filter4_3", shape=[3, 3, 512, 512]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv4_3"))
        conv4_3 = tf.Print(conv4_3, [tf.shape(conv4_3)], message="Conv4_3 shape:")

        # Max pooling2D
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='SAME', name="pool4")
        pool4 = tf.Print(pool4, [tf.shape(pool4)], message="Pool 4 shape:")

        # 3 x conv2D
        conv5_1 = tf.nn.relu(tf.nn.conv2d(pool4, filter=Vgg.initialize_variable(
                                            "filter5_1", shape=[3, 3, 512, 512]),
                              strides=[1, 1, 1, 1], padding='SAME', name="conv5_1"))
        conv5_1 = tf.Print(conv5_1, [tf.shape(conv5_1)], message="Conv5_1 shape:")

        conv5_2 = tf.nn.relu(tf.nn.conv2d(conv5_1, filter=Vgg.initialize_variable(
                                            "filter5_2", shape=[3, 3, 512, 512]),
                              strides=[1, 1, 1, 1], padding='SAME', name="conv5_2"))
        conv5_2 = tf.Print(conv5_2, [tf.shape(conv5_2)], message="Conv5_2 shape:")

        conv5_3 = tf.nn.relu(tf.nn.conv2d(conv5_2, filter=Vgg.initialize_variable(
                                            "filter5_3", shape=[3, 3, 512, 512]),
                              strides=[1, 1, 1, 1], padding='SAME', name="conv5_3"))
        conv5_3 = tf.Print(conv5_3, [tf.shape(conv5_3)], message="Conv5_3 shape:")

        # Max pooling2D
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                 padding='SAME', name="pool5")
        pool5 = tf.Print(pool5, [tf.shape(pool5)], message="Pool 5 shape:")

        # 3 x Dense
        fc6 = tf.nn.relu(tf.nn.conv2d(pool5, filter=Vgg.initialize_variable(
                                            "filter6", shape=[7, 7, 512, 4096]),
                              strides=[1, 1, 1, 1], padding='SAME', name="conv19"))
        fc6 = tf.Print(fc6, [tf.shape(fc6)], message="fc6 shape:")

        if self.is_encoder:

            return fc6

        else:

            fc7 = tf.nn.relu(tf.nn.conv2d(fc6, filter=Vgg.initialize_variable(
                                            "filter7", shape=[1, 1, 4096, 4096]),
                                strides=[1, 1, 1, 1], padding='SAME', name="conv20"))
            fc7 = tf.Print(fc7, [tf.shape(fc7)], message="fc7 shape:")

            fc8 = tf.nn.relu(tf.nn.conv2d(fc7, filter=Vgg.initialize_variable(
                                            "filter8", shape=[1, 1, 4096, dim_output]),
                                strides=[1, 1, 1, 1], padding='SAME', name="conv21"))
            fc8 = tf.Print(fc8, [tf.shape(fc8)], message="fc8 shape:")

            return fc8

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

        # Compute probabilities
        logit = tf.nn.softmax(tf.squeeze(self.model))

        # Loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit,
                                                          labels=self.label)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        # Initialize variables
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init_op)

            for epoch in range(self.n_epochs):

                for i in range(self.batch_size, len(training_set)):

                    batch_examples = training_set[i - self.batch_size: i]

                    image_batch, label_batch = self.load_batch(batch_examples)

                    _, cost = sess.run([optimizer, loss], feed_dict={
                        self.input: image_batch,
                        self.label: label_batch
                    })

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
        image = ComputerVision.load_image(image_path, grayscale=self.grayscale,
                                          binarize=self.binarize,
                                          normalize=self.normalize)

        label = np.zeros(self.n_classes)
        label[label_id] = 1

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
