import tensorflow as tf


class Vgg:

    def __init__(self, layers=16, batch_size=1, time_step=3, n_features=1, n_hidden_units=512,
                 n_layers=1, n_classes=1, learning_rate=10, n_epochs=1):
        """
        Initialization of the Vgg model.

        Args:
            layers: the number of layers
            batch_size: the size of batch
            time_step: the time step
            n_features: the number of features
            n_hidden_units: the number of hidden units
            n_layers: the number of layers
            n_classes: the number of classes
            learning_rate: the learning rate applied in the gradient descent optimization
            n_epochs: the number of epochs

        Returns:
            Nothing
        """
        tf.reset_default_graph()

        self.layers = layers
        self.batch_size = batch_size
        self.time_step = time_step
        self.n_features = n_features

        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.n_classes= n_classes

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # Input
        self.input = tf.placeholder(
            shape=(self.batch_size, self.time_step, self.n_features), dtype=tf.float32)

        # Label
        self.label = tf.placeholder(
            shape=(self.batch_size, self.n_classes), dtype=tf.float32)

        # Build model
        self.model = self.build_model()

        # Parameters
        weights = {
            'out': tf.Variable(tf.random_normal([self.n_hidden_units, self.n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

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

    def build_model(self):
        """
        Build the vgg graph model.

        Returns:
            tensorflow layer
        """

        # 2 x conv2D
        conv1_1 = tf.nn.conv2d(self.input, filter=Vgg.initialize_variable(
                                                    "filter1_1", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv1_1")
        conv1_1 = tf.Print(conv1_1, [tf.Shape(conv1_1)], message="Conv1_1 shape:")

        conv1_2 = tf.nn.conv2d(conv1_1, filter=Vgg.initialize_variable(
                                                    "filter1_2", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv1_2")
        conv1_2 = tf.Print(conv1_2, [tf.Shape(conv1_2)], message="Conv1_2 shape:")

        # Max pooling2D
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                                padding='SAME', name="pool1")
        pool1 = tf.Print(pool1, [tf.Shape(pool1)], message="Pool 1 shape:")

        # 2 x conv2D
        conv2_1 = tf.nn.conv2d(pool1, filter=Vgg.initialize_variable(
                                                    "filter2_1", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv2_1")
        conv2_1 = tf.Print(conv2_1, [tf.Shape(conv2_1)], message="Conv2_1 shape:")

        conv2_2 = tf.nn.conv2d(conv2_1, filter=Vgg.initialize_variable(
                                                    "filter2_2", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv2_2")
        conv2_2 = tf.Print(conv2_2, [tf.Shape(conv2_2)], message="Conv2_2 shape:")

        # Max pooling2D
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                                padding='SAME', name="pool2")
        pool2 = tf.Print(pool2, [tf.Shape(pool2)], message="Pool 2 shape:")

        # 3 x conv2D
        conv3_1 = tf.nn.conv2d(pool2, filter=Vgg.initialize_variable(
                                                    "filter3_1", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv3_1")
        conv3_1 = tf.Print(conv3_1, [tf.Shape(conv3_1)], message="conv3_1 shape:")

        conv3_2 = tf.nn.conv2d(conv3_1, filter=Vgg.initialize_variable(
                                                    "filter3_2", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv3_2")
        conv3_2 = tf.Print(conv3_2, [tf.Shape(conv3_2)], message="conv3_2 shape:")

        conv3_3 = tf.nn.conv2d(conv3_2, filter=Vgg.initialize_variable(
                                                    "filter3_3", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv3_3")
        conv3_3 = tf.Print(conv3_3, [tf.Shape(conv3_3)], message="Conv3_3 shape:")

        # Max pooling2D
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                                padding='SAME', name="pool3")
        pool3 = tf.Print(pool3, [tf.Shape(pool3)], message="Pool 3 shape:")

        # 3 x conv2D
        conv4_1 = tf.nn.conv2d(pool3, filter=Vgg.initialize_variable(
                                                    "filter4_1", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv4_1")
        conv4_1 = tf.Print(conv4_1, [tf.Shape(conv4_1)], message="Conv4_1 shape:")

        conv4_2 = tf.nn.conv2d(conv4_1, filter=Vgg.initialize_variable(
                                                    "filter4_2", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv4_2")
        conv4_2 = tf.Print(conv4_2, [tf.Shape(conv4_2)], message="Conv4_2 shape:")

        conv4_3 = tf.nn.conv2d(conv4_2, filter=Vgg.initialize_variable(
                                                    "filter4_3", shape=[1, 3, 3, 1]),
                             strides=[1, 1, 1, 1], padding='SAME', name="conv4_3")
        conv4_3 = tf.Print(conv4_3, [tf.Shape(conv4_3)], message="Conv4_3 shape:")

        # Max pooling2D
        pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                                 padding='SAME', name="pool4")
        pool4 = tf.Print(pool4, [tf.Shape(pool4)], message="Pool 4 shape:")

        # 3 x conv2D
        conv5_1 = tf.nn.conv2d(pool4, filter=Vgg.initialize_variable(
                                                    "filter5_1", shape=[1, 3, 3, 1]),
                              strides=[1, 1, 1, 1], padding='SAME', name="conv5_1")
        conv5_1 = tf.Print(conv5_1, [tf.Shape(conv5_1)], message="Conv5_1 shape:")

        conv5_2 = tf.nn.conv2d(conv5_1, filter=Vgg.initialize_variable(
                                                    "filter5_2", shape=[1, 3, 3, 1]),
                              strides=[1, 1, 1, 1], padding='SAME', name="conv5_2")
        conv5_2 = tf.Print(conv5_2, [tf.Shape(conv5_2)], message="Conv5_2 shape:")

        conv5_3 = tf.nn.conv2d(conv5_2, filter=Vgg.initialize_variable(
                                                    "filter5_3", shape=[1, 3, 3, 1]),
                              strides=[1, 1, 1, 1], padding='SAME', name="conv5_3")
        conv5_3 = tf.Print(conv5_3, [tf.Shape(conv5_3)], message="Conv5_3 shape:")

        # Max pooling2D
        pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1],
                                 padding='SAME', name="pool5")
        pool5 = tf.Print(pool5, [tf.Shape(pool5)], message="Pool 5 shape:")

        # 3 x Dense
        fc6 = tf.nn.conv2d(pool5, filter=Vgg.initialize_variable(
                                                    "filter6", shape=[1, 7, 7, 1]),
                              strides=[1, 1, 1, 1], padding='SAME', name="conv19")
        fc6 = tf.Print(fc6, [tf.Shape(fc6)], message="fc6 shape:")

        fc7 = tf.nn.conv2d(fc6, filter=Vgg.initialize_variable(
                                                    "filter7", shape=[1, 1, 1, 1]),
                            strides=[1, 1, 1, 1], padding='SAME', name="conv20")
        fc7 = tf.Print(fc7, [tf.Shape(fc7)], message="fc7 shape:")

        fc8 = tf.nn.conv2d(fc7, filter=Vgg.initialize_variable(
                                                    "filter8", shape=[1, 1, 1, 1]),
                            strides=[1, 1, 1, 1], padding='SAME', name="conv21")
        fc8 = tf.Print(fc8, [tf.Shape(fc8)], message="fc8 shape:")

        return fc8

    def fit(self, x, y):
        """
        Fit the model weights with input and labels.

        Args:
            x: input array
            y: label array
        Returns:
            Nothing
        """

        # Generate prediction
        outputs, states = tf.contrib.rnn.static_rnn(self.lstm_model, self.input, dtype=tf.float32)
        pred = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

        # Loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                                      labels=self.label))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate
                                              ).minimize(loss)

        with tf.session() as session:

            for seq_x, seq_y in self.get_sequences_from_dataset(x, y):

                _, cost = session.run([optimizer, loss], feed_dict={
                    self.input: seq_x,
                    self.label: seq_y
                })



    def predict(self, x):
        """
        Predict the output from input.

        Args:
            x: input array

        Returns:
            predictions array
        """
        pass
