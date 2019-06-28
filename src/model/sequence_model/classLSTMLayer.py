import tensorflow as tf
from time import time
import numpy as np

from sequence_model.classLayer import Layer
from sequence_model.classLSTMCell import LSTMCell


class LSTMLayer(Layer):

    def __init__(self, units_per_cell, is_sequence_output=True, return_states=False, time_steps=1, n_output=1,
                 f_out="identity", seed=42):
        """
        Initialization of LSTM layer

        Args:
            units_per_cell: the number of units per RNN Cell
            is_sequence_output: whether the model outputs a sequence
            return_states: whether to return the states
            n_output: the output dimension
            f_out: the activation function used as output of cell
        """
        Layer.__init__(self,)

        self.units_per_cell = units_per_cell
        self.n_cells = time_steps  # the number of RNN cells is equivalent to the length of the sequence
        self.time_steps = time_steps
        self.is_sequence_output = is_sequence_output
        self.return_states = return_states
        self.n_output = n_output
        self.f_out = f_out

        self.input_keep_prob = 1
        self.cell_keep_prob = 1
        self.output_keep_prob = 1
        self.y_keep_prob = 0.5
        self.seed = seed

    def build(self, input, initial_cell, initial_output=None, name="layer"):
        """
        Build

        Args:
            input: a tensor representing input (batch size x time steps x features)
            initial_cell: a tensor representing cell (batch size x number of units)
            initial_output: a tensor representing the initial output (batch size x number of output)
            name: the name of the variable scope

        Returns:
            the output tensor as a sequence or not
        """
        outputs = []
        with tf.variable_scope(name):

            cell_t = initial_cell
            output_t = initial_output

            for i in range(self.n_cells):

                input_t = input[:, i, :]

                if i < self.n_cells - 1 and not self.is_sequence_output:
                    lstm_cell = LSTMCell(units=self.units_per_cell, f_out=self.f_out, return_output=False,
                                         input_keep_prob=self.input_keep_prob, cell_keep_prob=self.cell_keep_prob,
                                         output_keep_prob=self.output_keep_prob, y_keep_prob=self.y_keep_prob,
                                         seed=self.seed + i)
                    output_t, cell_t = lstm_cell.build(input_t, cell_t, output_t, name="lstm_cell_" + str(i))

                else:
                    lstm_cell = LSTMCell(units=self.units_per_cell, f_out=self.f_out, return_output=True,
                                         input_keep_prob=self.input_keep_prob, cell_keep_prob=self.cell_keep_prob,
                                         output_keep_prob=self.output_keep_prob, y_keep_prob=self.y_keep_prob,
                                         seed=self.seed + i)

                    output_t, cell_t, y_t = lstm_cell.build(input_t, cell_t, output_t, name="lstm_cell_" + str(i))

                    if self.return_states:
                        outputs.append(output_t)
                    else:
                        outputs.append(y_t)

            return outputs
