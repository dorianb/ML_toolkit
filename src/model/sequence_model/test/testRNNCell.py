import unittest
import tensorflow as tf

from sequence_model.classRNNCell import RNNCell


class RNNCellTestCase(unittest.TestCase):

    def test_init(self):
        """
        Test the initialization method of RNN Cell.
        """
        rnn_cell_1 = RNNCell(n_unit=100, f_out=tf.nn.softmax, n_output=2,
                             with_prev_output=False, return_output=False)
        self.assertEqual(rnn_cell_1.n_unit, 100)

    def test_build(self):
        """
        Test the build method of RNN Cell.

        """
        batch_size = 2
        n_features = 10
        n_unit = 100
        n_output = 2
        input_t = tf.placeholder(tf.float32, shape=(batch_size, n_features))
        prev_state = tf.placeholder(tf.float32, shape=(batch_size, n_unit))
        prev_output = tf.placeholder(tf.float32, shape=(batch_size, n_output))

        rnn_cell_1 = RNNCell(n_unit=n_unit, f_out=tf.nn.softmax, n_output=n_output,
                             with_prev_output=False, return_output=False)
        state = rnn_cell_1.build(input_t, prev_state, name="rnn_cell_1")

        self.assertTupleEqual(tuple(state.get_shape().as_list()), (batch_size, n_unit))

        rnn_cell_1 = RNNCell(n_unit=n_unit, f_out=tf.nn.softmax, n_output=n_output,
                             with_prev_output=True, return_output=True)
        state, output = rnn_cell_1.build(input_t, prev_state, prev_output, name="rnn_cell_2")

        self.assertTupleEqual(tuple(state.get_shape().as_list()), (batch_size, n_unit))
        self.assertTupleEqual(tuple(output.get_shape().as_list()), (batch_size, n_output))

if __name__ == '__main__':
    unittest.main()
