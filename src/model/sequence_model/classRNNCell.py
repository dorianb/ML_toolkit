import tensorflow as tf
from sequence_model.classCell import Cell


class RNNCell(Cell):

    def __init__(self, n_unit, f_out, n_output=1, with_prev_output=True, return_output=True):
        """
        Initialize a RNN Cell.

        Args:
            n_unit: the number of hidden units
            f_out: the output activation function (softmax or identity)
            n_output: the dimension of output
            with_prev_output: whether to use the previous output in addition to the input
            return_output: whether to compute and return an output in addition to the state
        """
        Cell.__init__(self)
        self.n_unit = n_unit
        self.f_out = f_out
        self.n_output = n_output
        self.with_prev_output = with_prev_output
        self.return_output = return_output

    def build(self, input_t, prev_state, prev_output=None, name="rnn_cell", seed=42):
        """
        Build a RNN Cell graph.

        Args:
            input_t: a tensor input
            prev_state: the previous state
            prev_output: the previous output
            name: the scope name
            seed: the seed for randomized initializationsS

        Returns:
            a tuple with state and output if the last is computed
        """
        n_features = input_t.get_shape()[-1]
        prev_cell_unit = prev_state.get_shape()[-1]

        with tf.variable_scope(name_or_scope=name):

            w_input = self.initialize_variable(
                (n_features, self.n_unit), initializer="glorot_uniform",
                dtype=tf.float32, name="input_w", seed=seed)
            w_state = self.initialize_variable(
                (prev_cell_unit, self.n_unit), initializer="orthogonal",
                dtype=tf.float32, name="state_w", seed=seed)

            if self.with_prev_output and prev_output is not None:
                w_back = self.initialize_variable(
                    (self.n_output, self.n_unit), initializer="random_uniform",
                    dtype=tf.float32, name="back_w", seed=seed
                )

            b_x = self.initialize_variable(
                (self.n_unit, ), initializer="zeros", name="b_x",
                dtype=tf.float32, seed=seed)

            if self.return_output:
                w_output = self.initialize_variable(
                    (n_features + self.n_unit, self.n_output), initializer="glorot_uniform",
                    dtype=tf.float32, name="output_w", seed=seed)
                b_y = self.initialize_variable(
                    (self.n_output,), initializer="zeros",
                    dtype=tf.float32, name="b_output", seed=seed)

        with tf.name_scope(name=name):

            if self.with_prev_output and prev_output is not None:
                state_t = tf.nn.softsign(
                    tf.matmul(input_t, w_input) +
                    tf.matmul(prev_state, w_state) +
                    tf.matmul(prev_output, w_back) +
                    b_x
                )
            else:
                state_t = tf.nn.softsign(
                    tf.matmul(input_t, w_input) +
                    tf.matmul(prev_state, w_state) +
                    b_x
                )

            if self.return_output:
                output_t = self.f_out(
                    tf.matmul(tf.concat([input_t, state_t], axis=1), w_output) +
                    b_y
                )

                return state_t, output_t

            else:
                return state_t



