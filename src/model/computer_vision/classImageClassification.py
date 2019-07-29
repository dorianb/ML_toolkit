import numpy as np
from time import time
import tensorflow as tf

from dataset_utils.functionImageUtils import load_image
from computer_vision.classComputerVision import ComputerVision


class ImageClassification(ComputerVision):

    def __init__(self):
        """
        The initialization of a computer vision model.

        Args:
            summary_path: the path to the summaries
            checkpoint_path: the path to the checkpoints
        """
        ComputerVision.__init__(self)

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

    def fit(self, training_set, validation_set):
        """
        Fit the model weights with input and labels.

        Args:
            training_set: the training input and label
            validation_set: the validation input and label

        Returns:
            Nothing
        """

        if self.is_encoder:
            raise Exception("Vgg Fit method is implemented for image classification "
                            "purpose only")

        # Loss
        loss = ComputerVision.compute_loss(self.model, self.label,
                                           loss_name="softmax_cross_entropy")

        # Compute probabilities
        logit = tf.nn.softmax(self.model)
        logit = tf.Print(logit, [logit], message="Probabilities: ",
                         summarize=self.n_classes * self.batch_size) if self.debug else logit

        # Accuracy
        accuracy = ComputerVision.compute_accuracy(self, logit, self.label)

        # Optimization
        train_op = ComputerVision.compute_gradient(self, loss, self.global_step)

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
            ComputerVision.load(self, sess) if self.from_pretrained else None

            self.global_step = tf.add(self.global_step, tf.constant(1))

            for epoch in range(self.n_epochs):

                for i in range(self.batch_size, len(training_set)+self.batch_size, self.batch_size):

                    time0 = time()
                    batch_examples = training_set[i - self.batch_size: i]

                    image_batch, label_batch = self.load_batch(batch_examples)

                    _, loss_value, summaries_value, accuracy_value, step = sess.run([
                        train_op, loss, summaries, accuracy, self.global_step],
                        feed_dict={
                            self.input: image_batch,
                            self.label: label_batch
                        }
                    )

                    self.logger.info("Writing summary to {0}".format(self.summary_path)) if self.logger else None
                    self.train_writer.add_summary(summaries_value, step)

                    time1 = time()
                    self.logger.info(
                        "Accuracy = {0}, Cost = {1} for batch {2} of epoch {3} in {4:.2f} seconds".format(
                            accuracy_value, loss_value, i / self.batch_size, epoch, time1 - time0)) if self.logger else None

                    if i % self.validation_step == 0:

                        self.validation_eval(sess, summaries,
                                             validation_set[:self.validation_size],
                                             step)

                    if i % self.checkpoint_step == 0:

                        ComputerVision.save(self, sess, step=self.global_step)

    def load_batch(self, examples, with_labels=True):
        """
        Load the batch examples.

        Args:
            examples: the example in the batch
            with_labels: whether to return label
        Returns:
            the batch examples
        """
        images = []
        labels = []
        for example in examples:
            if with_labels:
                image, label = self.load_example(example)
                images.append(image)
                labels.append(label)
            else:
                image = self.load_example(example, with_labels=with_labels)
                images.append(image)

        if with_labels:
            return np.stack(images), np.stack(labels)
        else:
            return np.stack(images)

    def load_example(self, example, with_labels=True):
        """
        Load the example.

        Args:
            example: an example with image path and label
            with_labels: whether to return label
        Returns:
            the example image array and label
        """
        if with_labels:
            image_path, label_id = example
            self.logger.info("Loading example: {0} with label {1}".format(
                image_path, label_id)) if self.logger else None

        else:
            image_path = example
            self.logger.info("Loading example: {0}".format(
                image_path)) if self.logger else None

        image = load_image(image_path, grayscale=self.grayscale,
                                          binarize=self.binarize,
                                          normalize=self.normalize,
                                          resize_dim=self.resize_dim)
        if with_labels:
            label = np.zeros(self.n_classes)
            label[int(label_id)] = 1

            return image, label
        else:
            return image

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
        images, labels = self.load_batch(dataset)

        summaries_value = session.run(
            summaries,
            feed_dict={
                self.input: images,
                self.label: labels
            }
        )

        self.validation_writer.add_summary(summaries_value, step)

    def predict(self, dataset):
        """
        Predict the output from input.

        Args:
            dataset: the input dataset

        Returns:
            predictions array
        """

        # Compute probabilities
        logit = tf.nn.softmax(self.model)

        # Get predictions
        pred = tf.argmax(logit, axis=-1)

        with tf.Session() as sess:

            # Load existing model
            ComputerVision.load(self, sess)

            predictions = []

            for i in range(self.batch_size, len(dataset)+self.batch_size, self.batch_size):

                batch_examples = dataset[i - self.batch_size: i]

                image_batch = self.load_batch(batch_examples, with_labels=False)

                pred_batch = sess.run(
                    pred,
                    feed_dict={
                        self.input: image_batch
                    }
                )

                predictions += pred_batch.tolist()

            return predictions
