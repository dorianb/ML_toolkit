import unittest
import tensorflow as tf
import numpy as np

from sequence_model.classRNN import RNN


class RNNTestCase(unittest.TestCase):

    def test_init(self):
        """
        Initialize a RNN

        """
        units = [
            [100, 50, 10]
        ]
        f_out = tf.identity
        batch_size = 2

        rnn_1 = RNN(units, f_out, batch_size, n_output=1, with_prev_output=False,
                 with_input=True, return_sequences=True)

    def test_build_model(self):
        """
        Build RNN model.

        """
        units = [
            [100, 50, 10, 50, 100]
        ]
        f_out = tf.identity
        batch_size = 2
        time_steps = 5
        n_features = 10
        n_output = 3

        input_seq = tf.placeholder(
            shape=(batch_size, time_steps, n_features), dtype=tf.float32
        )

        rnn_1 = RNN(
            units, f_out, batch_size, n_output=n_output, with_prev_output=False,
            with_input=True, return_sequences=True
        )

        prediction = rnn_1.build_model(input_seq, name="rnn_1")

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, time_steps, n_output))

        rnn_2 = RNN(
            units, f_out, batch_size, n_output=n_output, with_prev_output=False,
            with_input=True, return_sequences=False
        )

        prediction = rnn_2.build_model(input_seq, name="rnn_2")

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, n_output))

        rnn_3 = RNN(
            units, f_out, batch_size, n_output=n_output, with_prev_output=False,
            with_input=False, return_sequences=False
        )

        prediction = rnn_3.build_model(input_seq, name="rnn_3")

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, n_output))

        units = [
            [100, 50, 10, 50, 100],
            [200, 100, 20, 100, 200],
            [300, 150, 30, 150, 300]
        ]

        rnn_4 = RNN(
            units, f_out, batch_size, n_output=n_output, with_prev_output=True,
            with_input=True, return_sequences=True
        )

        prediction = rnn_4.build_model(input_seq, name="rnn_4")

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, time_steps, n_output))

        rnn_5 = RNN(
            units, f_out, batch_size, n_output=n_output, with_prev_output=True,
            with_input=False, return_sequences=False
        )

        prediction = rnn_5.build_model(input_seq, name="rnn_5")

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, n_output))

        rnn_6 = RNN(
            units, f_out, batch_size, n_output=n_output, with_prev_output=False,
            with_input=False, return_sequences=False
        )

        prediction = rnn_6.build_model(input_seq, name="rnn_6")

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, n_output))


if __name__ == '__main__':
    unittest.main()