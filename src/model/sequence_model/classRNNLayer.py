import tensorflow as tf


from sequence_model.classLayer import Layer
from sequence_model.classRNNCell import RNNCell


class RNNLayer(Layer):

    def __init__(self, units_per_cell, is_sequence_output=True, return_states=False, with_prev_output=True,
                 time_steps=1, n_output=1, f_out="identity", seed=42):
        """
        Initialization of RNN layer

        Args:
            units_per_cell: the number of units per RNN Cell
            is_sequence_output: whether the model outputs a sequence
            return_states: whether to return the states
            with_prev_output: whether the model uses the previous cell output
            n_output: the output dimension
            f_out: the activation function used as output of cell
        """
        Layer.__init__(self, )

        self.units_per_cell = units_per_cell
        self.is_sequence_output = is_sequence_output
        self.return_states = return_states
        self.with_prev_output = with_prev_output
        self.n_cells = time_steps
        self.n_output = n_output
        self.f_out = f_out

        self.input_keep_prob = 0.8
        self.state_keep_prob = 0.8
        self.output_keep_prob = 0.8

        self.seed = seed

    def build(self, input, initial_state, initial_output=None, name="layer"):
        """
        Build

        Args:
            input: a tensor representing input (batch size x time steps x features)
            initial_state: a tensor representing state (batch size x number of units)
            initial_output: a tensor representing the initial output (batch size x number of outputs)
            name: the name of the variable scope

        Returns:
            the output tensor as a sequence or not
        """
        outputs = []
        with tf.variable_scope(name):

            output = None
            state_t = initial_state
            # state_t = tf.nn.dropout(state_t, keep_prob=self.state_keep_prob)  # A ajouter lorsque l'état initial est une variable

            for i in range(self.n_cells):

                input_t = input[:, i, :]

                if i < self.n_cells - 1 and not self.is_sequence_output:
                    rnn_cell = RNNCell(units=self.units_per_cell, f_out=self.f_out, return_output=False,
                                       with_prev_output=self.with_prev_output, input_keep_prob=self.input_keep_prob,
                                       state_keep_prob=self.state_keep_prob, output_keep_prob=self.output_keep_prob,
                                       seed=self.seed + i)
                    state_t = rnn_cell.build(input_t, state_t, name="rnn_cell_" + str(i))

                else:
                    rnn_cell = RNNCell(units=self.units_per_cell, f_out=self.f_out, return_output=True,
                                       with_prev_output=self.with_prev_output, input_keep_prob=self.input_keep_prob,
                                       state_keep_prob=self.state_keep_prob, output_keep_prob=self.output_keep_prob,
                                       n_output=self.n_output, seed=self.seed + i)
                    if self.with_prev_output:
                        output, state_t = rnn_cell.build(input_t, state_t, output, name="rnn_cell_" + str(i))
                    else:
                        output, state_t = rnn_cell.build(input_t, state_t, name="rnn_cell_" + str(i))

                    if self.return_states:
                        outputs.append(state_t)
                    else:
                        outputs.append(output)

            return outputs
