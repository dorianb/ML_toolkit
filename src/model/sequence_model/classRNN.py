import tensorflow as tf
import numpy as np
from time import time
import os

from sequence_model.classSequenceModel import SequenceModel
from sequence_model.classRNNCell import RNNCell


class RNN(SequenceModel):

    def __init__(self, units, f_out, batch_size=2, time_steps=24, n_features=10, n_output=1,
                 with_prev_output=False, with_input=True, return_sequences=True,
                 n_epochs=1, validation_step=10, checkpoint_step=100, from_pretrained=False,
                 optimizer_name="rmsprop", learning_rate=0.001, loss_name='mse',
                 summary_path=".", checkpoint_path=".", name="rnn", logger=None, debug=False):
        """
        Initialize an RNN model.
        Args:
            units: a 2-dimensional list with units number for each layers' cells
            f_out: the activation function for the output
            batch_size: number of examples by batch
            time_steps: the number of time steps
            n_features: the number of input features
            n_output: the dimension of the output tensor
            with_prev_output: whether to use the previous output for the next cell
            with_input: whether to use an input at each stage or use the previous instead
            return_sequences: whether to return an ouput for each time step of sequence
            n_epochs: the number of epochs
            validation_step: the number of batch training between each evaluation
            checkpoint_step: the number of batch training between each checkpoints
            from_pretrained: whether to load pre trained model
            optimizer_name: the name of the optimizer
            learning_rate: the learning rate used by the optimizer
            loss_name: the name of the loss to minimize
            summary_path: the path of the summary
            checkpoint_path: the path to the model checkpoint
            logger: the logging instance to trace
            name: the scope name
            debug: debug mode
        """
        SequenceModel.__init__(self)

        self.units = np.array(units)
        self.f_out = f_out
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.n_features = n_features
        self.n_output = n_output
        self.with_prev_output= with_prev_output
        self.with_input = with_input
        self.return_sequences = return_sequences
        self.n_epochs = n_epochs
        self.validation_step = validation_step
        self.checkpoint_step = checkpoint_step
        self.from_pretrained = from_pretrained
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.loss_name = loss_name
        self.summary_path = os.path.join(summary_path, name)
        self.checkpoint_path = os.path.join(checkpoint_path, name)
        self.name = name
        self.logger = logger
        self.debug = debug

        # Input
        self.input = tf.placeholder(
            shape=(None, self.time_steps, self.n_features), dtype=tf.float32)

        # Label
        self.label = tf.placeholder(shape=(
            None, self.time_steps, self.n_output
        ) if self.return_sequences else (None, self.n_output), dtype=tf.float32)

        self.initial_states = None
        self.initial_outputs = None

        # Global step
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Optimizer
        self.optimizer = SequenceModel.get_optimizer(self.optimizer_name, self.learning_rate)

        # Summary writers
        self.train_writer, self.validation_writer = SequenceModel.get_writer(self)

        # Model saver
        self.saver = tf.train.Saver()

    def build_model(self, input_seq):
        """
        Build RNN graph.

        Args:
            input_seq: the input sequence tensor (batch, time step, features)

        Returns:
            the model output
        """
        with tf.variable_scope(name_or_scope=self.name):

            n_layers, n_cells = self.units.shape
            prev_layers_outputs = []

            self.initial_states = [tf.placeholder(
                name="initial_state_" + str(l), shape=(None, self.units[l][0]), dtype=tf.float32
            ) for l in range(n_layers)]

            self.initial_outputs = tf.placeholder(
                name="initial_outputs", shape=(n_layers, None, self.n_output), dtype=tf.float32
            ) if self.with_prev_output else None

            for l, layer_units in enumerate(self.units):

                with_prev_output = self.with_prev_output if l == 0 and self.with_input else False
                layer_outputs = []

                with tf.variable_scope(name_or_scope="layer_" + str(l)):

                    for t, cell_unit in enumerate(layer_units):

                        if t == 0:
                            prev_state = self.initial_states[l]
                            prev_output = None

                        if t == 0 and with_prev_output:
                            prev_output = self.initial_outputs[l]

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
            [-1, n_cells, self.n_output]
        ) if self.return_sequences else prev_output

    def fit(self, train_set, validation_set, initial_states, initial_outputs=None):
        """
        Fit model using training set.

        Args:
            train_set: the data set used for training
            validation_set: the data set used for evaluation
            initial_states: the initial states of the graph model
            initial_outputs: the initial outputs of the graph model

        """
        # Build the graph model
        output = self.build_model(self.input)

        # Loss
        loss = SequenceModel.compute_loss(output, self.label, loss=self.loss_name)

        # Optimization
        train_op = self.compute_gradient(loss, self.global_step)

        # Merge summaries
        summaries = tf.summary.merge_all()

        # Initialize variables
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()

        with tf.Session() as sess:

            sess.run(init_g)
            sess.run(init_l)

            self.train_writer.add_graph(sess.graph)

            # Load existing model
            SequenceModel.load(self, sess) if self.from_pretrained else None

            for epoch in range(self.n_epochs):

                for i in range(self.batch_size, len(train_set), self.batch_size):

                    self.global_step = tf.add(self.global_step, tf.constant(1))

                    time0 = time()
                    batch_examples = train_set[i - self.batch_size: i]

                    feature_batch, label_batch = SequenceModel.load_batch(batch_examples)

                    feed_dict = {
                        self.input: feature_batch,
                        self.label: label_batch
                    }
                    feed_dict.update({
                        i: np.repeat(d, self.batch_size, axis=0)
                        for i, d in zip(self.initial_states, initial_states)
                    })
                    feed_dict.update({
                        self.initial_outputs: np.repeat(
                            initial_outputs, self.batch_size, axis=0
                            ).reshape(initial_outputs.shape[0], self.batch_size, initial_outputs.shape[1])
                    }) if self.with_prev_output and initial_outputs is not None else None

                    _, loss_value, summaries_value, step = sess.run([
                        train_op, loss, summaries, self.global_step],
                        feed_dict=feed_dict
                    )

                    self.train_writer.add_summary(summaries_value, step)

                    time1 = time()
                    self.logger.info(
                        "Cost = {0} for batch {1} in {2:.2f} seconds".format(
                            loss_value, i / self.batch_size, time1 - time0)) if self.logger else None

                    if i % self.validation_step == 0:
                        self.validation_eval(sess, summaries, validation_set, initial_states, initial_outputs, loss, step)

                    if i % self.checkpoint_step == 0:
                        SequenceModel.save(self, sess, step=self.global_step)

    def validation_eval(self, session, summaries, dataset, initial_states, initial_outputs, loss, step):
        """
        Produce evaluation on the validation dataset.

        Args:
            session: the session object opened
            summaries: the summaries declared in the graph
            dataset: the dataset to use for validation
            initial_states: the initial states of the graph model
            initial_outputs: the initial outputs of the graph model
            loss: the tensorflow operation used to compute the loss
            step: the step of summarize writing

        Returns:
            Nothing
        """
        time0 = time()
        inputs, labels = SequenceModel.load_batch(dataset)

        feed_dict = {
            self.input: inputs,
            self.label: labels
        }
        feed_dict.update({
            i: np.repeat(d, len(dataset), axis=0)
            for i, d in zip(self.initial_states, initial_states)
        })
        feed_dict.update({
            self.initial_outputs: np.repeat(
                initial_outputs, len(dataset), axis=0
            ).reshape(
                initial_outputs.shape[0], len(dataset), initial_outputs.shape[1]
            )
        }) if self.with_prev_output and initial_outputs is not None else None

        loss_value, summaries_value = session.run([loss, summaries], feed_dict=feed_dict)

        self.validation_writer.add_summary(summaries_value, step)

        time1 = time()

        self.logger.info(
            "Cost = {0} for evaluation set of size {1} in {2:.2f} seconds".format(
                loss_value, len(dataset), time1 - time0)) if self.logger else None

    def predict(self, dataset):
        pass