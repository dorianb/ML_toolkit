import os
from datetime import datetime
import tensorflow as tf
import logging
import numpy as np
import psutil
from collections import OrderedDict


class SequenceModel:

    def __init__(self, name, debug):
        """
        The initialization of a sequential model.
        Args:
            name: name of the model
            debug: debug mode
        """
        self.summary_path = ""
        self.checkpoint_path = ""
        self.optimizer = None
        self.global_step = None
        self.saver = None
        self.name = name

        self.logger = logging.Logger(name, level=logging.DEBUG if debug else logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(name + ".log")
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

    def build_model(self):
        pass

    @staticmethod
    def memory():
        """
        Print memory usage
        """
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / (1024.0 ** 3)
        print('memory use:', memoryUse)

    @staticmethod
    def get_parameter(shape, initializer, name, seed=42):
        """
        Get the parameter like weight or bias.

        Args:
            shape: shape of tensor as tuple
            initializer: initializer of tensor as string
            name: name of the parameter as string
            seed: the seed for randomized initializers

        Returns:
            tensor
        """

        if initializer == "uniform":
            initializer = tf.random_uniform_initializer(seed=seed)
        elif initializer == "normal":
            initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=seed)
        elif initializer == "orthogonal":
            initializer = tf.orthogonal_initializer()
        elif initializer == "glorot_uniform":
            initializer = tf.glorot_uniform_initializer()
        elif initializer == "zeros":
            initializer = tf.zeros_initializer()
        else:
            initializer = tf.zeros_initializer()

        variable = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer)
        return variable

    @staticmethod
    def compute_loss(prediction, label, loss_name="mse", seq_length=1):
        """
        Compute the loss

        Args:
            prediction: the tensor of predictions (bach_size, n_outputs)
            label: the tensor of labels (batch_size, n_outputs)
            loss_name: string representing the name of the metric to use as loss
            seq_length: the sequence length
        Returns:
            scalar value representing loss
        """
        if loss_name.lower() == "mse":
            return SequenceModel.compute_mse(prediction, label, seq_length, name="Loss_"+loss_name)
        elif loss_name.lower() == "mae":
            return SequenceModel.compute_mae(prediction, label, seq_length, name="Loss_"+loss_name)
        elif loss_name.lower() == "mape":
            return SequenceModel.compute_mape(prediction, label, seq_length, name="Loss_"+loss_name)
        elif loss_name.lower() == "correlation":
            return SequenceModel.compute_correlation(prediction, label, seq_length, as_op=True, name="Loss_"+loss_name)
        else:
            return SequenceModel.compute_mse(prediction, label,seq_length, name="Loss_default")

    @staticmethod
    def compute_metrics(prediction, label, seq_length=1):
        """
        Compute the metrics

        Args:
            prediction: the tensor of predictions (bach_size, n_outputs) or (batch_size, seq_length, n_outputs)
            label: the tensor of labels (batch_size, n_outputs) or (batch_size, seq_length, n_outputs)
            seq_length: the sequence length
        Returns:
            scalar value representing correlation coefficient
        """
        mse = SequenceModel.compute_mse(prediction, label, seq_length)
        mae = SequenceModel.compute_mae(prediction, label, seq_length)
        mape = SequenceModel.compute_mape(prediction, label, seq_length)
        corr = SequenceModel.compute_correlation(prediction, label, seq_length, as_op=False)
        r2 = SequenceModel.compute_r2(prediction, label, seq_length)

        return OrderedDict({
            "mse": mse,
            "mae": mae,
            "mape": mape,
            "corr": corr,
            "r2": r2
        })

    @staticmethod
    def compute_correlation(prediction, label, seq_length=1, as_op=True, name='Correlation_coefficient'):
        """
        Compute the correlation

        Args:
            prediction: the tensor of predictions (bach_size, n_outputs) or (batch_size, seq_length, n_outputs)
            label: the tensor of labels (batch_size, n_outputs) or (batch_size, seq_length, n_outputs)
            seq_length: the sequence length
            as_op: whether to return operation or variable
            name: string representing the name of the variable
        Returns:
            scalar value representing correlation coefficient
        """
        if seq_length == 1:
            corr, corr_op = tf.contrib.metrics.streaming_pearson_correlation(prediction, label)
        else:
            corr, corr_op = tf.reduce_mean(
                [tf.contrib.metrics.streaming_pearson_correlation(prediction[t], label[:, t, :])[int(~as_op)]
                 for t in range(seq_length)], axis=0)

        tf.summary.scalar(name, corr_op)
        return corr_op

    @staticmethod
    def compute_mse(prediction, label, seq_length=1, name="Mean_squared_error"):
        """
        Compute the mean absolute percentage error

        Args:
            prediction: the tensor of predictions (bach_size, n_outputs) or (batch_size, seq_length, n_outputs)
            label: the tensor of labels (batch_size, n_outputs) or (batch_size, seq_length, n_outputs)
            seq_length: the sequence length
            name: string representing the name of the variable
        Returns:
            scalar value representing mse
        """
        if seq_length == 1:
            mse = tf.reduce_mean(tf.square(prediction - label), axis=[0, 1])
        else:
            mse = tf.reduce_mean(
                [tf.reduce_mean(tf.square(prediction[t], label[:, t, :]), axis=[0, 1])
                 for t in range(seq_length)], axis=0)

        tf.summary.scalar(name, mse)
        return mse

    @staticmethod
    def compute_mae(prediction, label, seq_length=1, name="Mean_absolute_error"):
        """
        Compute the mean absolute error

        Args:
            prediction: the tensor of predictions (bach_size, n_outputs) or (batch_size, seq_length, n_outputs)
            label: the tensor of labels (batch_size, n_outputs) or (batch_size, seq_length, n_outputs)
            seq_length: the sequence length
            name: string representing the name of the variable
        Returns:
            scalar value representing mse
        """
        if seq_length == 1:
            mae = tf.reduce_mean(tf.abs(prediction - label), axis=[0, 1])
        else:
            mae = tf.reduce_mean(
                [tf.reduce_mean(tf.abs(prediction[t], label[:, t, :]), axis=[0, 1])
                 for t in range(seq_length)], axis=0)

        tf.summary.scalar(name, mae)
        return mae

    @staticmethod
    def compute_r2(prediction, label, seq_length=1, name="R2"):
        """
        Compute the squared determination coefficient
        Args:
            prediction:  the tensor of predictions (bach_size, n_outputs) or (batch_size, seq_length, n_outputs)
            label: the tensor of labels (batch_size, n_outputs) or (batch_size, seq_length, n_outputs)
            seq_length: the sequence length
            name: string representing the name of the variable
        Returns:
            scalar value representing r2
        """
        if seq_length == 1:
            r2 = tf.reduce_mean(tf.divide(
                tf.reduce_sum(tf.square(prediction - tf.reduce_mean(label, axis=0)), axis=0),
                tf.reduce_sum(tf.square(label - tf.reduce_mean(label, axis=0)), axis=0)
            ))
        else:
            r2 = tf.reduce_mean(

                [tf.reduce_mean(tf.divide(
                    tf.reduce_sum(tf.square(prediction[t] - tf.reduce_mean(label[:, t, :], axis=0)), axis=0),
                    tf.reduce_sum(tf.square(label[:, t, :] - tf.reduce_mean(label[:, t, :], axis=0)), axis=0)
                )) for t in range(seq_length)], axis=0)

        tf.summary.scalar(name, r2)
        return r2

    @staticmethod
    def compute_mape(prediction, label, seq_length=1, name="Mean_absolute_percentage_error"):
        """
        Compute the mean absolute percentage error

        Args:
            prediction:  the tensor of predictions (bach_size, n_outputs) or (batch_size, seq_length, n_outputs)
            label: the tensor of labels (batch_size, n_outputs) or (batch_size, seq_length, n_outputs)
            seq_length: the sequence length
            name: string representing the name of the variable
        Returns:
            scalar value representing mape
        """
        if seq_length == 1:
            mape = tf.reduce_mean(
                tf.multiply(
                    tf.divide(tf.abs(prediction - label), tf.abs(label)),
                    tf.constant(100.0)
                ),
                axis=[0, 1]
            )
        else:
            mape = tf.reduce_mean(
                [tf.reduce_mean(
                    tf.multiply(
                        tf.divide(tf.abs(prediction[t] - label[:, t, :]), tf.abs(label[:, t, :])),
                        tf.constant(100.0)
                    ),
                    axis=[0, 1]
                )
                 for t in range(seq_length)], axis=0)

        tf.summary.scalar(name, mape)
        return mape

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

    def fit(self, training_set, validation_set):
        pass

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
            input, label = example
            input = input.reshape(1, -1) if len(input.shape) == 1 else input
            inputs.append(input)
            labels.append(label)

        return np.stack(inputs), np.stack(labels)

    def load_example(self, example):
        pass

    @staticmethod
    def variable_summaries(var, name):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

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

    @staticmethod
    def get_activation(name="identity"):
        """
        Get the activation function.

        Args:
            name: name of the activation function as string

        Returns:
            tensorflow operation

        """
        if name == "identity":
            return tf.identity
        elif name == "relu":
            return tf.nn.relu
        elif name == "softmax":
            return tf.nn.softmax
        else:
            return tf.identity

    def get_writer(self):
        """
        Get the training and validation summaries writers.
        Returns:
            Tensorflow FileWriter object
        """
        training_path = os.path.join(
            self.summary_path, "train", str(datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        )
        validation_path = os.path.join(
            self.summary_path, "val", str(datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        )
        return tf.summary.FileWriter(training_path), tf.summary.FileWriter(validation_path)

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
        step = session.run(self.global_step)
        self.logger.info("Loaded model from %s at step %d" % (filename, step)) if self.logger else None

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

    def validation_eval(self, session, summaries, dataset, loss, step):
        pass

    def predict(self, dataset):
        pass
