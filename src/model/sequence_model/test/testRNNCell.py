import unittest
import tensorflow as tf

from sequence_model.classRNNCell import RNNCell


class RNNCellTestCase(unittest.TestCase):

    def test_build(self):
        """
        Test the build method

        """
        units = 100
        batch_size = 2
        p = 10
        n_output = 4

        input = tf.placeholder(shape=(batch_size, p), dtype=tf.float32)
        state = tf.get_variable(shape=(batch_size, units), initializer=tf.random_uniform_initializer(),
                                dtype=tf.float32,
                                name="initial_state")
        prev_output = tf.get_variable(shape=(batch_size, n_output), initializer=tf.random_uniform_initializer(),
                                      dtype=tf.float32, name="prev_output")

        rnn_cell_1 = RNNCell(units=units, f_out=tf.nn.softmax, return_output=True, with_prev_output=False, n_output=n_output)

        output, state = rnn_cell_1.build(input, state, name="rnn_cell_1")

        self.assertTupleEqual(tuple(output.get_shape().as_list()), (batch_size, n_output))
        self.assertTupleEqual(tuple(state.get_shape().as_list()), (batch_size, units))

        rnn_cell_2 = RNNCell(units=units, f_out=tf.nn.softmax, return_output=False, with_prev_output=False, n_output=n_output)

        state = rnn_cell_2.build(input, state, name="rnn_cell_2")

        self.assertTupleEqual(tuple(state.get_shape().as_list()), (batch_size, units))

        rnn_cell_3 = RNNCell(units=units, f_out=tf.nn.softmax, return_output=True, with_prev_output=True, n_output=n_output)

        output, state = rnn_cell_3.build(input, state, prev_output, name="rnn_cell_3")

        self.assertTupleEqual(tuple(output.get_shape().as_list()), (batch_size, n_output))
        self.assertTupleEqual(tuple(state.get_shape().as_list()), (batch_size, units))


if __name__ == '__main__':
    unittest.main()
