import os
import shutil

from dataset_utils.Dataset import Dataset


class ImageClassificationDataset(Dataset):

    def __init__(self, training_path, testing_path=None, train_size=0.7, val_size=0.2, test_size=0.1, absolute_path=True):
        """
        Initialize a CaltechDataset instance object.

        Args:
            training_path: the path to the dataset folder used for training
            testing_path: the path to the dataset folder used for evaluation (no labels)
            train_size: ratio of training dataset used for training
            val_size: ratio of training dataset used for validation
            test_size: ratio of training dataset used for testing
            absolute_path: whether to used absolute or relative path
        Returns:
            Nothing
        """
        Dataset.__init__(self)

        self.training_path = training_path
        self.testing_path = testing_path
        self.labels = self.get_labels()

        self.training_examples = self.get_training_examples(absolute_path)
        self.testing_examples = self.get_testing_examples(absolute_path) if self.testing_path else []

        self.training_set, self.validation_set, self.test_set = self.train_val_test(
            self.training_examples, train_size=train_size, val_size=val_size, test_size=test_size)

    def get_labels(self):
        """
        Get the labels from subdirectories' names

        Returns:
            the labels
        """
        return {i: label
                for i, label in enumerate(sorted(os.listdir(self.training_path)))
                if os.path.isdir(os.path.join(self.training_path, label))
                }

    def get_training_examples(self, absolute_path):
        """
        Get the images from the dataset folder and their label.

        Args:
            absolute_path: boolean specifying whether to store absolute or relative path
        Returns:
            list of tuple with image path, label index
        """
        result = []

        for index, label in self.labels.items():

            label_path = os.path.join(self.training_path, label)

            for image_filename in os.listdir(label_path):

                image_path = os.path.join(label_path, image_filename) if absolute_path else os.path.join(label, image_filename)
                result.append((image_path, index))

        return result

    def get_testing_examples(self, absolute_path):
        """
        Get the images from the dataset folder.

        Args:
            absolute_path: boolean specifying whether to store absolute or relative path
        Returns:
            list of tuple with image path, label index
        """
        result = []

        for image_filename in os.listdir(self.testing_path):

            image_path = os.path.join(self.testing_path, image_filename) if absolute_path else os.path.join(self.testing_path, image_filename)
            result.append(image_path)

        return result

    def save_test_pred(self, predictions, directory):
        """
        Save the test set predictions into folders with labels'names.
        Args:
            predictions: list of predictions
            directory: the directory path where to save the images
        Returns:
            Nothing
        """
        for image, label in zip(self.testing_examples, predictions):

            # create the directory if it does not exist
            dir_path = os.path.join(directory, str(label))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            # copy the image to the new directory
            filename = os.path.basename(image)
            new_path = os.path.join(dir_path, filename)
            shutil.copy(image, new_path)
