import tensorflow as tf


class LSTM:

    def __init__(self, batch_size=1, time_step=3, n_features=1, n_hidden_units=512,
                 n_layers=1, n_classes=1, learning_rate=10, n_epochs=1):
        """
        Initialization of the LSTM model.

        Args:
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

        self.batch_size = batch_size
        self.time_step = time_step
        self.n_features = n_features

        self.n_hidden_units = n_hidden_units
        self.n_layers = n_layers
        self.n_classes= n_classes

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        # Input
        input = tf.placeholder(
            shape=(self.batch_size, self.time_step, self.n_features), dtype=tf.float32)

        # Label
        self.label = tf.placeholder(
            shape=(self.batch_size, self.n_classes), dtype=tf.float32)

        # Build model
        self.lstm_model = self.model(self.n_hidden_units, self.n_layers)

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

    def lstm_layer(self, n_hidden_units, name_scope="lstm_layer"):
        """
        Create the graph of a lstm layer.

        Args:
            n_hidden_units: the number of hidden units
            name_scope: the scope name of variables

        returns:
            tensorflow operation
        """

        """
        with tf.variable_scope(name_or_scope=name_scope, reuse=True):
            c = self.memory_cell()
            o = self.output_gate()
            return tf.matmul(o, tf.tanh(c))
        """
        return tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units, forget_bias=1.0,
            state_is_tuple=True, activation=None, reuse=None, name=None)

    def input_gate(self):
        """
        Create the input gate graph.

        Args:

        Returns:

        """
        weight_xi = LSTM.initialize_variable("weight_xi", shape=(self.n_features))
        weight_hi = LSTM.initialize_variable("weight_hi", shape=(2))
        weight_ci = LSTM.initialize_variable("weight_ci", shape=())

    def output_gate(self):
        """
        Create the output gate graph.

        Args:

        Returns:

        """
        pass

    def memory_cell(self, num_units=1):
        """
        Create the memory cell graph.

        Args:

        Returns:

        """
        pass

    def forget_gate(self):
        """
        Create the forget gate graph.

        Args:

        Returns:

        """
        pass

    def model(self, n_hidden_units, n_layers):
        """
        Build the LSTM graph model.

        Returns:
            tensorflow layer
        """
        rnn_cells = [self.lstm_layer(n_hidden_units, name_scope="lstm_layer_{0}".format(i))
                     for i in range(n_layers)]

        return tf.contrib.rnn.MultiRNNCell(rnn_cells)

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
