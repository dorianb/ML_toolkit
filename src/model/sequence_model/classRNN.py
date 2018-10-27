import tensorflow as tf
import numpy as np
from time import time

from sequence_model.classSequenceModel import SequenceModel
from sequence_model.classRNNCell import RNNCell


class RNN(SequenceModel):

    def __init__(self, units, f_out, batch_size, time_steps, n_features, n_output,
                 with_prev_output=False, with_input=True, return_sequences=True,
                 n_epochs=1, validation_step=10, from_pretrained=False,
                 optimizer_name="rmsprop", summary_path=".", checkpoint_path=".",
                 name="rnn", logger=None, debug=False):
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
            validation_step: the number of batch training between each checkpoints
            from_pretrained: whether to load pre trained model
            optimizer_name: the name of the optimizer
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
        self.validatin_step = validation_step
        self.from_pretrained = from_pretrained
        self.optimizer_name = optimizer_name
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
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

        # Global step
        self.global_step = tf.Variable(0, dtype=tf.int32, name="global_step")

        # Optimizer
        self.optimizer = SequenceModel.get_optimizer(self.optimizer_name, self.learning_rate)

        # Summary writers
        self.train_writer, self.validation_writer = SequenceModel.get_writer(self)

        # Model saver
        self.saver = tf.train.Saver()

    def build_model(self, input_seq, name="rnn"):
        """
        Build RNN graph.

        Args:
            input_seq: the input sequence tensor (batch, time step, features)
            name: the scope name

        Returns:
            the model output
        """
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

    def fit(self, train_set, validation_set):
        """
        Fit model using training set.

        Args:
            train_set: the data set used for training
            validation_set: the data set used for evaluation

        """
        # Build the graph model
        output = self.build_model(self.input, name=self.name)

        # Loss
        loss = SequenceModel.compute_loss(output, self.label)

        # Optimization
        train_op = SequenceModel.compute_gradient(loss, self.global_step)

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

                    image_batch, label_batch = self.load_batch(batch_examples)

                    _, loss_value, summaries_value, step = sess.run([
                        train_op, loss, summaries, self.global_step],
                        feed_dict={
                            self.input: image_batch,
                            self.label: label_batch
                        }
                    )

                    self.train_writer.add_summary(summaries_value, step)

                    time1 = time()
                    self.logger.info(
                        "Cost = {0} for batch {1} in {2:.2f} seconds".format(
                            loss_value, i / self.batch_size, time1 - time0)) if self.logger else None

                    if i % self.validation_step == 0:
                        self.validation_eval(sess, summaries, validation_set, step)

                    if i % self.checkpoint_step == 0:
                        SequenceModel.save(self, sess, step=self.global_step)

    def validation_eval(self, session, summaries, dataset, step):
        """
        Produce evaluation on the validation dataset.

        Args:
            session: the session object opened
            summaries: the summaries declared in the graph
            dataset: the dataset to use for validation
            step: the step of summarize writing

        Returns:
            Nothing
        """
        inputs, labels = SequenceModel.load_batch(dataset)

        feed_dict = {
            self.input: inputs,
            self.label: labels
        }

        feed_dict['']

        summaries_value = session.run(summaries, feed_dict=feed_dict)

        self.validation_writer.add_summary(summaries_value, step)

    def predict(self, dataset):
        pass