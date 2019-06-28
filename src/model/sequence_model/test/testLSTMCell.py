import unittest
import tensorflow as tf

from sequence_model.classLSTMCell import LSTMCell


class LSTMCellTestCase(unittest.TestCase):

    def test_build(self):
        """
        Test the build method

        """
        units = 100
        batch_size = 2
        p = 10
        n_output = 4

        input = tf.placeholder(shape=(batch_size, p), dtype=tf.float32)
        prev_cell = tf.get_variable(shape=(batch_size, units), initializer=tf.random_uniform_initializer(),
                                    dtype=tf.float32, name="prev_cell")
        prev_output = tf.get_variable(shape=(batch_size, units), initializer=tf.random_uniform_initializer(),
                                      dtype=tf.float32, name="prev_output")

        lstm_cell_1 = LSTMCell(units=units, f_out=tf.nn.softmax, return_output=True, n_output=n_output)

        output, cell, y = lstm_cell_1.build(input, prev_cell, prev_output, name="lstm_cell_1")

        self.assertTupleEqual(tuple(output.get_shape().as_list()), (batch_size, units))
        self.assertTupleEqual(tuple(cell.get_shape().as_list()), (batch_size, units))
        self.assertTupleEqual(tuple(y.get_shape().as_list()), (batch_size, n_output))

        lstm_cell_2 = LSTMCell(units=units, f_out=tf.nn.softmax, return_output=False, n_output=n_output)

        output, cell = lstm_cell_2.build(input, prev_cell, prev_output, name="lstm_cell_2")

        self.assertTupleEqual(tuple(output.get_shape().as_list()), (batch_size, units))
        self.assertTupleEqual(tuple(cell.get_shape().as_list()), (batch_size, units))


if __name__ == '__main__':
    unittest.main()
