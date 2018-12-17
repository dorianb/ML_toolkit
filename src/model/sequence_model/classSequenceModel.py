import tensorflow as tf
import numpy as np
import os
from datetime import datetime


class SequenceModel:

    def __init__(self):
        """
        Initialize a sequential model.
        """
        tf.reset_default_graph()
        self.checkpoint_path = None
        self.summary_path = None
        self.global_step = None
        self.name = None
        self.debug = None
        self.logger = None
        self.saver = None
        self.optimizer = None
        self.n_output = None

    def build_model(self, input_seq, name="sequence_model"):
        pass

    @staticmethod
    def get_optimizer(name="adam", learning_rate=0.1):
        """
        Get the optimizer object corresponding. If unknown optimizer, raise an exception.

        Args:
            name: name of the optimizer
            learning_rate: the learning rate
        Returns:
            a tensorflow optimizer object
        """
        if name == "adam":
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif name == "adadelta":
            return tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        elif name == "gradientdescent":
            return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif name == "rmsprop":
            return tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        else:
            raise Exception("The optimizer is unknown")

    def get_writer(self):
        """
        Get the training and validation summaries writers.

        Returns:
            Tensorflow FileWriter object
        """
        training_path = os.path.join(self.summary_path, "train", str(datetime.now()))
        validation_path = os.path.join(self.summary_path, "val", str(datetime.now()))
        return tf.summary.FileWriter(training_path), \
            tf.summary.FileWriter(validation_path)

    def load(self, session):
        """
        Load the model variables values.

        Args:
            session: the tensorflow session

        Returns:
            Nothing
        """
        step = sorted([
            int(filename.split(self.name)[1].split("-")[1].split(".")[0])
            for filename in os.listdir(self.checkpoint_path)
            if self.name in filename
        ]).pop()
        filename = self.name + "-" + str(step)
        checkpoint_path = os.path.join(self.checkpoint_path, filename)
        self.saver.restore(session, checkpoint_path)
        self.global_step = self.global_step.assign(step)
        step = session.run(self.global_step)
        self.logger.info("Loaded model from %s at step %d" % (filename, step)
                         ) if self.logger else None

    def save(self, session, step):
        """
        Persist the model variables values.

        Args:
            session: the tensorflow session
            step: the global step as a tensor

        Returns:
            the path to the saved model
        """
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        checkpoint_path = os.path.join(self.checkpoint_path, self.name)
        return self.saver.save(session, checkpoint_path, global_step=step)

    @staticmethod
    def load_batch(examples):
        """
        Load the batch examples.

        Args:
            examples: the example in the batch

        Returns:
            the batch examples
        """
        inputs = []
        labels = []
        for example in examples:
            features, label = example
            inputs.append(features)
            labels.append(label)

        return np.stack(inputs), np.stack(labels)

    def sample(self, p):
        """
        Sample an output from the probabilities estimates (used in classification).

        Args:
            p: a tensor of class probabilities (batch x nb classes)

        Returns:
            the  of the sampled prediction
        """
        elems = tf.convert_to_tensor(range(self.n_output))
        samples = tf.multinomial(tf.log(p), 1)
        indices = elems[tf.cast(samples, tf.int32)]
        return tf.one_hot(indices, depth=self.n_output, on_value=1.0, off_value=0.0, axis=-1)

    @staticmethod
    def compute_loss(output, label, loss='mse'):
        """
        Compute the loss operation.

        Args:
            output: the output of the model class probabilities or prediction
            label: the tensor of labels
            loss: the loss name

        Returns:
            loss: the loss
        """
        if loss == 'cross_entropy':
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=output, labels=label
            ))
        elif loss == 'mse':
            loss = tf.reduce_mean(tf.square(output - label))
        else:
            loss = tf.reduce_mean(tf.square(output - label))

        tf.summary.scalar('Loss', loss)
        return loss

    def compute_accuracy(self, logit, label):
        """
        Compute the accuracy measure.

        Args:
            logit: the tensor of class probabilities (bach_size, n_classes)
            label: the tensor of labels (batch_size, n_classes)

        Returns:
            accuracy: the accuracy metric measure

        """
        pred = tf.argmax(logit, axis=-1)
        y = tf.argmax(label, axis=-1)

        pred = tf.Print(pred, [pred], message="Prediction: ",
                        summarize=2) if self.debug else pred
        y = tf.Print(y, [y], message="Label: ",
                     summarize=2) if self.debug else y

        accuracy = tf.reduce_mean(tf.cast(tf.equal(y, pred), "float"))
        tf.summary.scalar('Accuracy', accuracy)

        return accuracy

    def compute_gradient(self, loss, global_step, max_value=1.):
        """
        Compute gradient and update parameters.

        Args:
            loss: the loss to minimize
            global_step: the training step
            max_value: the gradients max value
        Returns:
            the training operation
        """
        gvs = self.optimizer.compute_gradients(loss)
        self.logger.debug(gvs) if self.logger else None
        capped_gvs = [
            (
                tf.clip_by_value(grad, -max_value, max_value) if grad is not None else grad,
                var
            )
            for grad, var in gvs
        ]
        return self.optimizer.apply_gradients(capped_gvs, global_step=global_step)

    def fit(self, train_set, validation_set):
        """
        Fit model using training set.

        Args:
            train_set: the data set used for training
            validation_set: the data set used for evaluation
        """
        pass

    def predict(self):
        pass
