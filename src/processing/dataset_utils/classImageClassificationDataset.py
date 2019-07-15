import os
from dataset_utils.Dataset import Dataset


class ImageClassificationDataset(Dataset):

    def __init__(self, path, train_size=0.7, val_size=0.2, test_size=0.1, absolute_path=True):
        """
        Initialize a CaltechDataset instance object.

        Args:
            path: the path to the dataset folder
        Returns:
            Nothing
        """
        Dataset.__init__(self)

        self.dataset_path = path
        self.labels = self.get_labels()

        images_and_labels = self.get_examples(absolute_path)

        self.training_set, self.validation_set, self.test_set = self.train_val_test(
            images_and_labels, train_size=train_size, val_size=val_size,
            test_size=test_size)

    def get_labels(self):
        """
        Get the labels from subdirectories' names

        Returns:
            the labels
        """
        return {i: label
                for i, label in enumerate(sorted(os.listdir(self.dataset_path)))}

    def get_examples(self, absolute_path):
        """
        Get the images from the dataset folder and their label.

        Args:
            absolute_path: boolean specifying whether to store absolute or relative path
        Returns:
            list of tuple with image path, label index
        """
        result = []

        for index, label in self.labels.items():

            label_path = os.path.join(self.dataset_path, label)

            for image_filename in os.listdir(label_path):

                image_path = os.path.join(label_path, image_filename) if absolute_path else os.path.join(label, image_filename)
                result.append((image_path, index))

        return result
