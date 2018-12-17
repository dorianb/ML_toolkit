import argparse
import logging
import os
import traceback
import pandas as pd

from dataset_utils.classImageClassificationDataset import ImageClassificationDataset

parser = argparse.ArgumentParser(description='Dataset building program')
parser.add_argument('--dataset-path', type=str, help='Path to the dataset', default=".")
parser.add_argument('--train-size', type=float, help='Training set size', default=0.7)
parser.add_argument('--validation-size', type=float, help='Validation set size', default=0.2)
parser.add_argument('--test-size', type=float, help='Test set size', default=0.1)
parser.add_argument('--name', type=str, help='The unique name of the program', default="")
parser.add_argument('--debug', type=int, help='Debug mode', default=0)
args = parser.parse_args()

logger = logging.Logger("dataset_building_exec",
                        level=logging.DEBUG if args.debug else logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG if args.debug else logging.INFO)
logger.addHandler(consoleHandler)

fileHandler = logging.FileHandler("dataset_building_exec.log")
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)

assert os.path.isdir(args.dataset_path), "{0} is not a valid directory".format(args.dataset_path)

try:

    classes_path = os.path.join(args.dataset_path, "classes.csv")
    training_set_path = os.path.join(args.dataset_path, "training_set.csv")
    validation_set_path = os.path.join(args.dataset_path, "validation_set.csv")
    test_set_path = os.path.join(args.dataset_path, "test_set.csv")

    os.remove(classes_path)
    os.remove(training_set_path)
    os.remove(validation_set_path)
    os.remove(test_set_path)

    dataset_1 = ImageClassificationDataset(
        args.dataset_path, train_size=args.train_size,
        val_size=args.validation_size, test_size=args.test_size,
        absolute_path=False)

    classes = dataset_1.labels
    classes.update({0: 'ambiguous'})

    df_classes = pd.DataFrame.from_dict(classes, orient='index')
    df_training = pd.DataFrame(dataset_1.training_set)
    df_val = pd.DataFrame(dataset_1.training_set)
    df_test = pd.DataFrame(dataset_1.training_set)

    df_classes.to_csv(classes_path, header=False, index=True)
    df_training.to_csv(training_set_path, header=False, index=False)
    df_val.to_csv(validation_set_path, header=False, index=False)
    df_test.to_csv(test_set_path, header=False, index=False)

except Exception:
    logger.error(traceback.format_exc())
