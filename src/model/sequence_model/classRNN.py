import tensorflow as tf
from time import time
import numpy as np
import gc

from sequence_model.classSequenceModel import SequenceModel
from sequence_model.classRNNCell import RNNCell


class RNN(SequenceModel):

    def __init__(self, units_per_cell, batch_size, time_steps, n_features, n_layers=1, is_sequence_output=True,
                 with_prev_output=True, n_output=1, f_out="identity", optimizer_name="rmsprop", learning_rate=0.1,
                 loss_name="mse", epochs=1, from_pretrained=False, validation_step=1000, checkpoint_step=1000,
                 summary_path="", checkpoint_path="", name="rnn", debug=0):
        """
        Initialization of RNN model

        Args:
            units_per_cell: the number of units per RNN Cell
            batch_size: the size of batch
            time_steps: the length of sequence
            n_features: the number of features in the sequence
            n_layers: the number of layers stacked
            is_sequence_output: whether the model outputs a sequence
            with_prev_output: whether the model uses the previous cell output
            n_output: the output dimension
            f_out: the activation function used as output of cell
            optimizer_name: the optimizer name used
            learning_rate: the learning rate
            loss_name: the name of the metric to use for optimization
            epochs: the number of epochs for training
            from_pretrained: whether to use the pre trained model
            validation_step: model evaluation frequency
            checkpoint_step: model checkpoint frequency
            summary_path: the path to the summary folder
            checkpoint_path: the path to the checkpoint path
            name: the name of the variable scope
            debug: whether debug mode is activate
        """
        tf.reset_default_graph()

        SequenceModel.__init__(self, name, debug)

        self.units_per_cell = units_per_cell
        self.n_layers = n_layers
        self.n_cells = time_steps  # the number of RNN cells is equivalent to the length of the sequence
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.n_features = n_features
        self.is_sequence_output = is_sequence_output
        self.with_prev_output = with_prev_output
        self.n_output = n_output
        self.loss_name = loss_name
        self.epochs = epochs
        self.from_pretrained = from_pretrained
        self.validation_step = validation_step
        self.checkpoint_step = checkpoint_step
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path

        self.input_keep_prob = 1
        self.state_keep_prob = 1
        self.output_keep_prob = 1
        self.seed = 42

        self.input = tf.placeholder(tf.float32, [None, self.time_steps, self.n_features])
        self.initial_state = tf.placeholder(tf.float32, [None, self.units_per_cell])

        if self.is_sequence_output:
            self.label = tf.placeholder(tf.float32, [None, self.time_steps, self.n_output])

        else:
            self.label = tf.placeholder(tf.float32, [None, self.n_output])

        # Global step
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Optimizer
        self.optimizer = self.get_optimizer(optimizer_name, learning_rate)

        # Output activation
        self.f_out = self.get_activation(f_out)

        # Summary writers
        self.train_writer, self.validation_writer = self.get_writer()

        # Model saver
        self.saver = tf.train.Saver()

        # Build model
        self.model = self.build_model(name)

    def build_model(self, name="rnn"):
        """
        Build the model

        Args:
            name: the name of the variable scope

        Returns:
            the output tensor as a sequence or not
        """
        outputs = []
        with tf.variable_scope(name):

            output = None
            state_t = self.initial_state
            # state_t = tf.nn.dropout(state_t, keep_prob=self.state_keep_prob)  # A ajouter lorsque l'état initial est une variable

            for i in range(self.n_cells):

                input_t = self.input[:, i, :]

                if i < self.n_cells - 1 and not self.is_sequence_output:
                    rnn_cell = RNNCell(units=self.units_per_cell, f_out=self.f_out, return_output=False,
                                       with_prev_output=False, input_keep_prob=self.input_keep_prob,
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

                    outputs.append(output)

            return outputs

    def fit(self, training_set, validation_set):
        """
        Fit the model

        Args:
            training_set:
            validation_set:

        Returns:

        """
        outputs = self.model

        outputs[-1] = tf.Print(outputs[-1], [self.label], summarize=self.batch_size * self.n_output, message="Label: ")
        outputs[-1] = tf.Print(outputs[-1], [outputs[-1]], summarize=self.batch_size * self.n_output,
                               message="Prediction: ")

        metrics = SequenceModel.compute_metrics(outputs[-1], self.label, self.time_steps if self.is_sequence_output else 1)
        loss = SequenceModel.compute_loss(outputs[-1], self.label, self.loss_name, self.time_steps if self.is_sequence_output else 1)

        train_op = SequenceModel.compute_gradient(self, loss, self.global_step)

        # Merge summaries
        summaries = tf.summary.merge_all()

        # Initialize variables
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        self.global_step = tf.add(self.global_step, tf.constant(1))

        with tf.Session() as sess:

            run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

            sess.run(init_g)
            sess.run(init_l)

            self.train_writer.add_graph(sess.graph)

            # Load existing model
            SequenceModel.load(self, sess) if self.from_pretrained else None

            for epoch in range(self.epochs):

                for i in range(self.batch_size, len(training_set), self.batch_size):

                    time0 = time()
                    batch_input, batch_label = self.load_batch(training_set[i - self.batch_size: i])
                    initial_state = np.zeros(shape=(self.batch_size, self.units_per_cell), dtype=np.float32)

                    _, loss_value, summaries_value, step = sess.run([
                        train_op, loss, summaries, self.global_step],
                        feed_dict={
                            self.input: batch_input,
                            self.label: batch_label,
                            self.initial_state: initial_state
                        },
                        options=run_opts,
                    )

                    self.train_writer.add_summary(summaries_value, step)

                    time1 = time()
                    self.logger.info(
                        "Cost = {0} for batch {1} in {2:.2f} seconds".format(
                            loss_value, i / self.batch_size, time1 - time0)) if self.logger else None

                    if i % self.validation_step == 0:
                        SequenceModel.memory()
                        self.validation_eval(sess, summaries, validation_set, metrics, step)
                        gc.collect()
                        SequenceModel.memory()

                    if i % self.checkpoint_step == 0:
                        SequenceModel.save(self, sess, step=self.global_step)

    def validation_eval(self, session, summaries, dataset, metrics, step):
        """
        Produce evaluation on the validation dataset.

        Args:
            session: the session object opened
            summaries: the summaries declared in the graph
            dataset: the dataset to use for validation
            metrics: dictionary of tensorflow operation to cumpute the metrics
            step: the step of summarize writing
        Returns:
            Nothing
        """
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        time0 = time()
        inputs, labels = self.load_batch(dataset)
        initial_state = np.zeros(shape=(len(dataset), self.units_per_cell), dtype=np.float32)

        values = session.run(
            list(metrics.values()) + [summaries],
            feed_dict={
                self.input: inputs,
                self.label: labels,
                self.initial_state: initial_state
            },
            options=run_opts
        )
        time1 = time()

        self.logger.info("{0} for evaluation os size {1} in {2:.2f} seconds".format(
            "; ".join([k + " = " + str(values[i]) for i, k in enumerate(metrics.keys())]),
            inputs.shape[0], time1 - time0)) if self.logger else None

        self.validation_writer.add_summary(values[-1], step)

    def predict(self, dataset):
        """
        Predict the output from input.

        Args:
            dataset: the input dataset
        Returns:
            predictions array
        """
        pass
