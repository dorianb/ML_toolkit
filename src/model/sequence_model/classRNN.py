import tensorflow as tf
import numpy as np

from sequence_model.classSequenceModel import SequenceModel
from sequence_model.classRNNCell import RNNCell


class RNN(SequenceModel):

    def __init__(self, units, f_out, batch_size, n_output=1, with_prev_output=False,
                 with_input=True, return_sequences=True):
        """
        Initialize an RNN model.
        Args:
            units: a 2-dimensional list with units number for each layers' cells
            f_out: the activation function for the output
            batch_size: number of examples by batch
            n_output: the dimension of the output tensor
            with_prev_output: whether to use the previous output for the next cell
            with_input: whether to use an input at each stage or use the previous instead
            return_sequences:
        """
        SequenceModel.__init__(self)

        self.units = np.array(units)
        self.f_out = f_out
        self.batch_size = batch_size
        self.n_output = n_output
        self.with_prev_output= with_prev_output
        self.with_input = with_input
        self.return_sequences = return_sequences

    def build_model(self, input_seq, name="rnn"):

        with tf.variable_scope(name_or_scope=name):

            n_layers, n_cells = self.units.shape
            prev_layers_outputs = []

            for l, layer_units in enumerate(self.units):

                with_prev_output = self.with_prev_output if l == 0 and self.with_input else False
                layer_outputs = []

                with tf.variable_scope(name_or_scope="layer_" + str(l)):

                    for t, cell_unit in enumerate(layer_units):

                        if t == 0:
                            prev_state = tf.placeholder(
                                name="initial_state", shape=(None, cell_unit),
                                dtype=tf.float32
                            )
                            prev_output = None

                        if (t == 0 and with_prev_output) or (t == 0 and not self.with_input):
                            prev_output = tf.placeholder(
                                name="initial_output", shape=(None, self.n_output),
                                dtype=tf.float32
                            )

                        input_t = input_seq[:, t, :] if self.with_input or t == 0 else prev_output
                        input_t = prev_layers_outputs[l-1][t] if l > 0 and self.with_input else input_t

                        return_output = (
                            self.return_sequences
                            or (not self.return_sequences and l < n_layers - 1)
                            or not self.with_input
                            or (not self.return_sequences and l == n_layers - 1 and t == n_cells - 1)
                        )

                        rnn_cell = RNNCell(cell_unit, self.f_out, n_output=self.n_output, with_prev_output=with_prev_output,
                                           return_output=return_output)

                        if return_output:
                            prev_state, prev_output = rnn_cell.build(input_t, prev_state, prev_output=prev_output, name="cell_" + str(t))
                            layer_outputs.append(prev_output)
                        else:
                            prev_state = rnn_cell.build(input_t, prev_state, prev_output=prev_output, name="cell_" + str(t))

                    prev_layers_outputs.append(layer_outputs)

        return tf.reshape(
            tf.concat(layer_outputs, axis=1),
            [self.batch_size, n_cells, self.n_output]
        ) if self.return_sequences else prev_output

    def fit(self):
        pass

    def predict(self):
        pass