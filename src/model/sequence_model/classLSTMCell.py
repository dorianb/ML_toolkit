import tensorflow as tf

from sequence_model.classCell import Cell


class LSTMCell(Cell):

    def __init__(self, units, f_out, return_output=True,
                 input_keep_prob=1, cell_keep_prob=1, output_keep_prob=1, y_keep_prob=1, n_output=1, seed=42):
        """
        Initialization of RNN Cell

        Args:
            units: integer representing the number of units
            f_out: the output activation function of the network (softmax, identity or relu)
            return_output: whether to compute output and return it
            input_keep_prob: the input keep probability for dropout layer
            cell_keep_prob: the cell keep probability for dropout layer
            output_keep_prob: the output keep probability for dropout layer
            y_keep_prob: the network output keep probability for dropout layer
            n_output: the number of output
            seed: the seed to use for randomized operations
        """
        super(Cell, self).__init__()

        self.units = units
        self.f_out = f_out
        self.state_size = units

        self.return_output = return_output

        self.input_keep_prob = input_keep_prob
        self.cell_keep_prob = cell_keep_prob
        self.output_keep_prob = output_keep_prob
        self.y_keep_prob = y_keep_prob

        self.n_output = n_output

        self.seed = seed

        self.input_x = None
        self.input_c = None
        self.input_h = None
        self.input_bias = None

        self.forget_x = None
        self.forget_c = None
        self.forget_h = None
        self.forget_bias = None

        self.cell_x = None
        self.cell_c = None
        self.cell_h = None
        self.cell_bias = None

        self.output_x = None
        self.output_c = None
        self.output_h = None
        self.output_bias = None

        self.y_kernel = None
        self.y_bias = None

    def input_gate(self, input_t, prev_output, prev_cell, name="input_gate"):
        """
        Compute input gate.

        Args:
            input_t: input tensor at time step t
            prev_output: output tensor at time step t-1
            prev_cell: cell tensor at time step t-1
            name: the name of the variable scope

        Returns:
            a tensor representing the output of input gate at time step t
        """
        input_shape = input_t.get_shape()

        with tf.variable_scope(name):

            self.input_x = Cell.get_parameter(shape=(input_shape[-1], self.units), initializer='glorot_uniform',
                                              name='input_kernel', seed=self.seed)
            self.input_h = Cell.get_parameter(shape=(self.units, self.units), initializer='orthogonal',
                                              name='output_kernel', seed=self.seed)
            self.input_c = Cell.get_parameter(shape=(self.units,), initializer='uniform',
                                              name='cell_peephole', seed=self.seed)

            self.input_bias = Cell.get_parameter(shape=(self.units,), initializer='zeros', name='bias',
                                                 seed=self.seed)

            return tf.sigmoid(
                tf.matmul(input_t, self.input_x) +
                tf.matmul(prev_output, self.input_h) +
                tf.multiply(prev_cell, self.input_c) +
                self.input_bias
            )

    def forget_gate(self, input_t, prev_output, prev_cell, name="forget_gate"):
        """
        Compute the forget gate.

        Args:
            input_t: input tensor at time step t
            prev_output: output tensor at time step t-1
            prev_cell: cell tensor at time step t-1
            name: the name of the variable scope

        Returns:
            a tensor representing the output of the forget gate
        """
        input_shape = input_t.get_shape()

        with tf.variable_scope(name):

            self.forget_x = Cell.get_parameter(shape=(input_shape[-1], self.units), initializer='glorot_uniform',
                                               name='input_kernel', seed=self.seed)
            self.forget_h = Cell.get_parameter(shape=(self.units, self.units), initializer='orthogonal',
                                               name='output_kernel', seed=self.seed)
            self.forget_c = Cell.get_parameter(shape=(self.units,), initializer='uniform',
                                               name='cell_peephole', seed=self.seed)

            self.forget_bias = Cell.get_parameter(shape=(self.units,), initializer='zeros', name='bias',
                                                  seed=self.seed)

            return tf.sigmoid(
                tf.matmul(input_t, self.forget_x) +
                tf.matmul(prev_output, self.forget_h) +
                tf.multiply(prev_cell, self.forget_c) +
                self.forget_bias
            )

    def memory_cell(self, forget_gate_t, input_gate_t, input_t, prev_output, prev_cell, name="memory_cell"):
        """
        Compute the memory cell.

        Args:
            forget_gate_t: forget gate tensor at time step t
            input_gate_t: input gate tensor at time step t
            input_t: input tensor at time step t
            prev_output: output tensor at time step t-1
            prev_cell: cell tensor at time step t-1
            name: the name of the variable scope

        Returns:
            a tensor representing the output of the forget gate
        """
        input_shape = input_t.get_shape()

        with tf.variable_scope(name):

            self.cell_x = Cell.get_parameter(shape=(input_shape[-1], self.units), initializer='glorot_uniform',
                                             name='input_kernel', seed=self.seed)
            self.cell_h = Cell.get_parameter(shape=(self.units, self.units), initializer='orthogonal',
                                             name='output_kernel', seed=self.seed)

            self.cell_bias = Cell.get_parameter(shape=(self.units,), initializer='zeros', name='bias',
                                                seed=self.seed)

            u = tf.nn.softsign(
                tf.matmul(input_t, self.cell_x) +
                tf.matmul(prev_output, self.cell_h) +
                self.cell_bias
            )

            return tf.multiply(forget_gate_t, prev_cell) + tf.multiply(input_gate_t, u)

    def output_gate(self, input_t, prev_output, cell_t, name="output_gate"):
        """
        Compute the output gate.

        Args:
            input_t: input tensor at time step t
            prev_output: output tensor at time step t-1
            prev_cell: cell tensor at time step t-1
            name: the name of the variable scope

        Returns:
            a tensor representing the output of the output gate
        """
        input_shape = input_t.get_shape()

        with tf.variable_scope(name):

            self.output_x = Cell.get_parameter(shape=(input_shape[-1], self.units), initializer='glorot_uniform',
                                               name='input_kernel', seed=self.seed)
            self.output_h = Cell.get_parameter(shape=(self.units, self.units), initializer='orthogonal',
                                               name='output_kernel', seed=self.seed)
            self.output_c = Cell.get_parameter(shape=(self.units,), initializer='uniform',
                                               name='cell_peephole', seed=self.seed)

            self.output_bias = Cell.get_parameter(shape=(self.units,), initializer='zeros', name='bias',
                                                  seed=self.seed)

            return tf.sigmoid(
                tf.matmul(input_t, self.output_x) +
                tf.matmul(prev_output, self.output_h) +
                tf.multiply(cell_t, self.output_c) +
                self.output_bias
            )

    def build(self, input_t, prev_cell, prev_output, name="lstm_cell"):
        """
        Build the LSTM cell.

        Args:
            input_t: input tensor for time step t
            prev_output: output tensor at time step t-1
            prev_cell: cell tensor at time step t-1
            name: the name of the variable scope

        Returns:
            tuple with output at time t, state at time t+1 and prediction at time t
        """
        input_t = tf.nn.dropout(input_t, keep_prob=self.input_keep_prob, seed=self.seed)

        with tf.variable_scope(name):

            self.y_kernel = Cell.get_parameter(shape=(self.units, self.n_output), initializer='uniform',
                                               name='y_kernel', seed=self.seed)

            self.y_bias = Cell.get_parameter(shape=(self.n_output,), initializer='uniform',
                                             name='y_bias', seed=self.seed)

            input_gate_t = self.input_gate(input_t, prev_output, prev_cell)
            forget_gate_t = self.forget_gate(input_t, prev_output, prev_cell)

            cell_gate_t = self.memory_cell(forget_gate_t, input_gate_t, input_t, prev_output, prev_cell)
            cell_gate_t = tf.nn.dropout(cell_gate_t, keep_prob=self.cell_keep_prob, seed=self.seed)

            output_gate_t = self.output_gate(input_t, prev_output, cell_gate_t)

            h_t = tf.multiply(output_gate_t, tf.nn.softsign(cell_gate_t))
            h_t = tf.nn.dropout(h_t, keep_prob=self.output_keep_prob, seed=self.seed)

            if self.return_output:
                y = self.f_out(tf.matmul(h_t, self.y_kernel) + self.y_bias)
                y = tf.nn.dropout(y, keep_prob=self.y_keep_prob, seed=self.seed)
                return h_t, cell_gate_t, y
            else:
                return h_t, cell_gate_t
