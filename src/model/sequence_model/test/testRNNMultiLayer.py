import unittest
import os
import numpy as np

from sequence_model.classRNNMultiLayer import RNNMultiLayer


ROOT_PATH = os.sep.join(os.path.normpath(os.getcwd()).split(os.path.sep)[:-2])
DATASET_PATH = os.path.join(ROOT_PATH, "data")
METADATA_PATH = os.path.join(ROOT_PATH, "metadata")


class RNNTestCase(unittest.TestCase):

    def test_init_build(self):
        """
        Test init and build model methods
        """
        batch_size = 3
        n_output = 1
        time_steps = 24
        units_per_cell = 100
        n_features = 10

        summary_path = os.path.join(METADATA_PATH, "summaries")
        checkpoint_path = os.path.join(METADATA_PATH, "checkpoints")

        name = "rnn_1"
        rnn_1 = RNNMultiLayer(
            units_per_cell, batch_size, time_steps, n_features, n_layers=1, is_sequence_output=True,
            with_prev_output=False, n_output=n_output, f_out="identity", optimizer_name="rmsprop",
            learning_rate=0.1, epochs=1, from_pretrained=False, validation_step=1000,
            checkpoint_step=1000, summary_path=os.path.join(summary_path, name),
            checkpoint_path=os.path.join(checkpoint_path, name), name=name)

        outputs = rnn_1.model
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), time_steps)
        self.assertTupleEqual(tuple(outputs[-1].get_shape().as_list()), (None, n_output))

        name = "rnn_2"
        rnn_2 = RNNMultiLayer(
            units_per_cell, batch_size, time_steps, n_features, n_layers=1, is_sequence_output=False,
            with_prev_output=False, n_output=n_output, f_out="identity", optimizer_name="rmsprop",
            learning_rate=0.1, epochs=1, from_pretrained=False, validation_step=1000,
            checkpoint_step=1000, summary_path=os.path.join(summary_path, name),
            checkpoint_path=os.path.join(checkpoint_path, name), name=name)

        outputs = rnn_2.model
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 1)
        self.assertTupleEqual(tuple(outputs[-1].get_shape().as_list()), (None, n_output))

    def test_fit(self):
        """
        Test the fit method
        """

        batch_size = 3
        n_output = 1
        time_steps = 24
        units_per_cell = 100
        n_features = 10

        summary_path = os.path.join(METADATA_PATH, "summaries")
        checkpoint_path = os.path.join(METADATA_PATH, "checkpoints")

        name = "rnn_1"
        rnn_1 = RNNMultiLayer(
            units_per_cell, batch_size, time_steps, n_features, n_layers=1, is_sequence_output=False,
            with_prev_output=False, n_output=n_output, f_out="identity",  optimizer_name="rmsprop",
            learning_rate=0.1, epochs=1, from_pretrained=False, validation_step=10,
            checkpoint_step=10, summary_path=os.path.join(summary_path, name),
            checkpoint_path=os.path.join(checkpoint_path, name), name=name, debug=1)

        n = 100
        random_sold_set = np.random.randint(low=1, high=100, size=n)
        dataset = [
            (
                np.stack([
                    np.linspace(sold_set * feature, sold_set * feature - time_steps, num=time_steps)
                    for feature in range(n_features)
                ], axis=0).T,
                np.array(sold_set + 1).reshape(1)
            )
            for sold_set in random_sold_set
        ]

        training_set, validation_set = dataset[:int(n*0.8)], dataset[int(n*0.8):]
        rnn_1.fit(training_set, validation_set)


if __name__ == '__main__':
    unittest.main()
