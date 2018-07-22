import numpy as np


class Dataset:

    def __init__(self):
        """
        Initialize a Dataset instance object

        """
        self.training_set = []
        self.validation_set = []
        self.test_set = []

    def train_val_test(self, examples, train_size=0.7, val_size=0.2,
                       test_size=0.1):
        """

        Args:
            examples: examples in the dataset
            train_size: the proportion of examples for the training set
            val_size: the proportion of examples for the validation_set
            test_size: the proportion of examples for the test_set

        Returns:

        """
        n = len(examples)
        indexes = np.arange(0, n)
        np.random.shuffle(indexes)

        last_train_index = int(train_size * n)
        train_indexes = indexes[0:last_train_index]

        last_val_index = last_train_index + int(val_size * n)
        val_indexes = indexes[last_train_index:last_val_index]

        last_test_index = last_val_index + int(test_size * n)
        test_indexes = indexes[last_val_index:last_test_index]

        print("Train indexes")
        examples = np.array(examples)
        return examples[train_indexes], examples[val_indexes], examples[test_indexes]

    def get_labels(self):
        """
        Get the labels of the dataset

        Returns:
            Dictionary with labels and index as key
        """
        pass

    def get_examples(self):
        """
        Get the examples with their label

        Returns:
            A list of tuple
        """
        pass