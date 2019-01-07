import os
from datetime import datetime
import tensorflow as tf


class ComputerVision:

    def __init__(self, summary_path="", checkpoint_path=""):
        """
        The initialization of a computer vision model.

        Args:
            summary_path: the path to the summaries
            checkpoint_path: the path to the checkpoints
        """
        tf.reset_default_graph()

        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
        self.global_step = None
        self.saver = None
        self.logger = None
        self.debug = None
        self.optimizer = None
        self.name = "computer_vision"

    def build_model(self):
        pass

    @staticmethod
    def get_initializer(name):
        """
        Get the initializer.

        Args:
            name: the name of the initializer

        Returns:
            A tensorflow initializer object instance
        """
        if name == "zeros":
            return tf.zeros_initializer()
        elif name == "uniform":
            return tf.random_uniform_initializer(minval=0, maxval=0.0001, seed=42)
        elif name == "normal":
            return tf.random_normal_initializer(mean=0.0, stddev=1.0, seed=42)
        else:
            return tf.zeros_initializer()

    @staticmethod
    def get_parameter(name, initializer_name, shape):
        """
        Initialize a variable

        Args:
            name: the name of the variable
            initializer_name: the name of the parameter initializing method
            shape: the shape of the variable

        Returns:
            tensorflow variable
        """
        initializer = ComputerVision.get_initializer(initializer_name)
        variable = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer)
        return variable

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

    def get_writer(self):
        """
        Get the training and validation summaries writers.

        Returns:
            Tensorflow FileWriter object
        """
        training_path = os.path.join(self.summary_path, "train", str(datetime.now()))
        validation_path = os.path.join(self.summary_path, "val", str(datetime.now()))
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
    def compute_soft_ncut(output, k, sigma_x):
        """
        Compute the soft normalized cut loss

        Args:
            output: a tensor representing an image with class probabilities for each pixel
            k: an integer representing the number of classes
            sigma_x: the scale of the spatial proximity
        Returns:
            the soft normalized cut loss tensorflow operation
        """

        """
        def cond():
            pass

        def body():
            pass



        weights = tf.while_loop(
            cond,
            body,
            tf.reshape(),
            shape_invariants=None,
            parallel_iterations=10,
            back_prop=True,
            swap_memory=False,
            name=None,
            maximum_iterations=None,
            return_same_structure=False
        )
        weights =
        loss = k - tf.reduce_sum(
                tf.reduce_sum() / tf.reduce_sum()
            )

        tf.summary.scalar("soft_n_cut", loss)
        return loss
        """

    @staticmethod
    def compute_loss(output, label, loss_name="sigmoid_cross_entropy"):
        """
        Compute the loss operation.

        Args:
            output: a tensor representing an output
            label: a tensor representing a label
            loss_name: the name of the loss to compute
        Returns:
            loss: the loss
        """
        if loss_name == "sigmoid_cross_entropy":
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=output, labels=label
            ))
        elif loss_name == "softmax_cross_entropy":
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=output, labels=label
            ))
        elif loss_name == 'mse':
            loss = tf.reduce_mean(tf.square(output - label))
        else:
            raise Exception("Unknown loss")

        tf.summary.scalar(loss_name, loss)
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

        pred = tf.Print(pred, [pred], message="Prediction: ", summarize=2) if self.debug else pred
        y = tf.Print(y, [y], message="Label: ", summarize=2) if self.debug else y

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

    def fit(self, training_set, validation_set):
        pass

    def load_batch(self, examples):
        pass

    def load_example(self, example):
        pass

    def validation_eval(self, session, summaries, dataset, step):
        pass

    def predict(self, dataset):
        pass

