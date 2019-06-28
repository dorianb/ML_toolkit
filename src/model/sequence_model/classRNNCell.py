import tensorflow as tf

from sequence_model.classCell import Cell


class RNNCell(Cell):

    def __init__(self, units, f_out, return_output=True, with_prev_output=True,
                 input_keep_prob=1, state_keep_prob=1, output_keep_prob=1, n_output=1, seed=42):
        """
        Initialization of RNN Cell

        Args:
            units: integer representing the number of units
            f_out: the activation function used in output of the network
            return_output: whether to compute output and return it
            with_prev_output: whether to use previous cell output
            input_keep_prob: the input keep probability for dropout layer
            state_keep_prob: the state keep probability for dropout layer
            output_keep_prob: the output keep probability for dropout layer
            n_output: the number of output
            seed: the seed to use for randomized operations
        """
        super(Cell, self).__init__()

        self.units = units
        self.state_size = units

        self.return_output = return_output
        self.with_prev_output = with_prev_output
        self.input_keep_prob = input_keep_prob
        self.state_keep_prob = state_keep_prob
        self.output_keep_prob = output_keep_prob
        self.n_output = n_output

        self.seed = seed

        # softsign activation function over tanh (it’s faster and less prone to saturation (~0 gradients)).
        self.f = tf.nn.softsign
        self.f_out = f_out

        self.input_kernel = None
        self.state_kernel = None
        self.back_kernel = None
        self.output_kernel = None

        self.state_bias = None
        self.output_bias = None

    def build(self, input_t, state_t, prev_output_t=None, name="rnn_cell"):
        """
        Build the RNN cell.

        Args:
            input_t: input tensor for time step t
            state_t: state tensor for time step t
            prev_output_t: the previous output
            name: the name of the variable scope

        Returns:
            tuple with output at time t and state at time t+1
        """
        input_shape = input_t.get_shape()

        input_t = tf.nn.dropout(input_t, keep_prob=self.input_keep_prob, seed=self.seed)

        with tf.variable_scope(name):

            self.input_kernel = Cell.get_parameter(shape=(input_shape[-1], self.units), initializer='glorot_uniform',
                                                   name='input_kernel', seed=self.seed)
            self.state_kernel = Cell.get_parameter(shape=(self.units, self.units), initializer='orthogonal',
                                                   name='state_kernel', seed=self.seed)
            self.back_kernel = Cell.get_parameter(shape=(self.n_output, self.units), initializer='uniform',
                                                  name='back_kernel', seed=self.seed)
            self.output_kernel = Cell.get_parameter(shape=(input_shape[-1] + self.units, self.n_output),
                                                    initializer='uniform', name='output_kernel_weight', seed=self.seed)

            self.state_bias = Cell.get_parameter(shape=(self.units,), initializer='zeros', name='state_bias',
                                                 seed=self.seed)
            self.output_bias = Cell.get_parameter(shape=(self.n_output,), initializer='zeros', name='output_bias',
                                                  seed=self.seed)

            h = tf.matmul(input_t, self.input_kernel)

            if self.with_prev_output and prev_output_t is not None:
                a = self.f(h + tf.matmul(state_t, self.state_kernel) + tf.matmul(prev_output_t, self.back_kernel) + self.state_bias)
            else:
                a = self.f(h + tf.matmul(state_t, self.state_kernel) + self.state_bias)

            a = tf.nn.dropout(a, keep_prob=self.state_keep_prob, seed=self.seed)

            if self.return_output:
                u = tf.concat([input_t, a], axis=1)
                o = self.f_out(tf.matmul(u, self.output_kernel) + self.output_bias)
                o = tf.nn.dropout(o, keep_prob=self.output_keep_prob, seed=self.seed)
                return o, a
            else:
                return a
