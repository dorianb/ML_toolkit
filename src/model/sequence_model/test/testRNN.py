import unittest
import tensorflow as tf
import numpy as np
import os

from sequence_model.classRNN import RNN


ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-4])
DATASET_PATH = os.path.join(ROOT_PATH, "data", "corpus")
SUMMARY_PATH = os.path.join(ROOT_PATH, "metadata", "summaries")
CHECKPOINT_PATH = os.path.join(ROOT_PATH, "metadata", "checkpoints")


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
                    with_input=True, return_sequences=True,
                    summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_1")

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

        rnn_1 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=False, with_input=True, return_sequences=True,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_1"
        )

        input_seq = tf.placeholder(shape=(batch_size, time_steps, n_features), dtype=tf.float32)

        prediction = rnn_1.build_model(input_seq)

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, time_steps, n_output))

        rnn_2 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=False, with_input=True, return_sequences=False,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_2"
        )

        input_seq = tf.placeholder(shape=(batch_size, time_steps, n_features), dtype=tf.float32)

        prediction = rnn_2.build_model(input_seq)

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, n_output))

        rnn_3 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=False, with_input=False, return_sequences=False,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_3"
        )

        input_seq = tf.placeholder(shape=(batch_size, time_steps, n_features), dtype=tf.float32)

        prediction = rnn_3.build_model(input_seq)

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, n_output))

        units = [
            [100, 50, 10, 50, 100],
            [200, 100, 20, 100, 200],
            [300, 150, 30, 150, 300]
        ]

        rnn_4 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=True, with_input=True, return_sequences=True,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_4"
        )

        input_seq = tf.placeholder(shape=(batch_size, time_steps, n_features), dtype=tf.float32)

        prediction = rnn_4.build_model(input_seq)

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, time_steps, n_output))

        rnn_5 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=True, with_input=False, return_sequences=False,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_5"
        )

        input_seq = tf.placeholder(shape=(batch_size, time_steps, n_features), dtype=tf.float32)

        prediction = rnn_5.build_model(input_seq)

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, n_output))

        rnn_6 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=False, with_input=False, return_sequences=False,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_6"
        )

        input_seq = tf.placeholder(shape=(batch_size, time_steps, n_features), dtype=tf.float32)

        prediction = rnn_6.build_model(input_seq)

        self.assertTupleEqual(tuple(prediction.get_shape().as_list()), (batch_size, n_output))

    def test_fit(self):
        """
        Test the fit method

        """
        units = [
            [100, 50, 10, 50, 100]
        ]
        f_out = tf.identity
        batch_size = 2
        time_steps = 5
        n_features = 10
        n_output = 3

        n_train = 100
        n_valid = 100

        train_set = [
            (np.random.rand(time_steps, n_features), np.ones((time_steps, n_output)) * i)
            for i in range(n_train)
        ]

        validation_set = [
            (np.random.rand(time_steps, n_features), np.ones((time_steps, n_output)) * i)
            for i in range(n_valid)
        ]

        initial_states = [np.random.rand(1, units[l][0]) for l in range(len(units))]

        rnn_1 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=False, with_input=True, return_sequences=True,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_1"
        )

        rnn_1.fit(train_set, validation_set, initial_states)

        train_set = [
            (np.random.rand(time_steps, n_features), np.ones(n_output) * i)
            for i in range(n_train)
        ]

        validation_set = [
            (np.random.rand(time_steps, n_features), np.ones(n_output) * i)
            for i in range(n_valid)
        ]

        rnn_2 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=False, with_input=True, return_sequences=False,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_2"
        )

        rnn_2.fit(train_set, validation_set, initial_states)

        rnn_3 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=False, with_input=False, return_sequences=False,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_3"
        )

        rnn_3.fit(train_set, validation_set, initial_states)

        units = [
            [100, 50, 10, 50, 100],
            [200, 100, 20, 100, 200],
            [300, 150, 30, 150, 300]
        ]

        train_set = [
            (np.random.rand(time_steps, n_features), np.ones((time_steps, n_output)) * i)
            for i in range(n_train)
        ]

        validation_set = [
            (np.random.rand(time_steps, n_features), np.ones((time_steps, n_output)) * i)
            for i in range(n_valid)
        ]

        initial_states = [np.random.rand(1, units[l][0]) for l in range(len(units))]
        initial_outputs = np.zeros((len(units), n_output))

        rnn_4 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=True, with_input=True, return_sequences=True,
            summary_path = SUMMARY_PATH, checkpoint_path = CHECKPOINT_PATH, name="rnn_4")

        rnn_4.fit(train_set, validation_set, initial_states, initial_outputs=initial_outputs)

        train_set = [
            (np.random.rand(time_steps, n_features), np.ones(n_output) * i)
            for i in range(n_train)
        ]

        validation_set = [
            (np.random.rand(time_steps, n_features), np.ones(n_output) * i)
            for i in range(n_valid)
        ]

        rnn_5 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=True, with_input=False, return_sequences=False,
            summary_path = SUMMARY_PATH, checkpoint_path = CHECKPOINT_PATH, name="rnn_5"
        )

        rnn_5.fit(train_set, validation_set, initial_states, initial_outputs=initial_outputs)

        rnn_6 = RNN(
            units, f_out, batch_size=batch_size, time_steps=time_steps, n_features=n_features,
            n_output=n_output, with_prev_output=False, with_input=False, return_sequences=False,
            summary_path=SUMMARY_PATH, checkpoint_path=CHECKPOINT_PATH, name="rnn_6"
        )

        rnn_6.fit(train_set, validation_set, initial_states)

if __name__ == '__main__':
    unittest.main()