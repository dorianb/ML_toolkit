import argparse
import logging
import os
import sys
import traceback

from computer_vision.classVgg import Vgg
from dataset_utils.CaltechDataset import CaltechDataset

"""
--dataset-path "/home/dorian/workspace/ML_toolkit/src/model/computer_vision/data/256_ObjectCategories" --train-size 0.7 --validation-size 0.2 --test-size 0.1 --train 1 --learning-rate 0.01 --debug 1 
"""

parser = argparse.ArgumentParser(description='Image classification program')
parser.add_argument('--dataset-path', type=str, help='Path to the dataset', default=".")
parser.add_argument('--train-size', type=float, help='Training set size', default=0.7)
parser.add_argument('--validation-size', type=float, help='Validation set size', default=0.2)
parser.add_argument('--test-size', type=float, help='Test set size', default=0.1)
parser.add_argument('--train', type=int, help='Training mode', default=1)
parser.add_argument('--learning-rate', type=float, help='Learning rate', default=0.01)
parser.add_argument('--debug', type=int, help='Debug mode', default=0)
args = parser.parse_args()

logger = logging.Logger("image_classification_exec",
                        level=logging.DEBUG if args.debug else logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG if args.debug else logging.INFO)
logger.addHandler(consoleHandler)

fileHandler = logging.FileHandler("image_classification_exec.log")
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)

assert os.path.isdir(args.dataset_path), "{0} is not a valid directory".format(args.dataset_path)

try:

    cd_1 = CaltechDataset(args.dataset_path, train_size=args.train_size,
                          val_size=args.validation_size, test_size=args.test_size)
    classes = cd_1.labels
    classes.update({0: 'ambiguous'})

    vgg_1 = Vgg(classes, batch_size=2, height=1200, width=800, dim_out=len(classes),
                grayscale=True, binarize=False, normalize=False,
                learning_rate=args.learning_rate, n_epochs=1, validation_step=10,
                is_encoder=False, validation_size=10, logger=logger, debug=args.debug)

    vgg_1.fit(cd_1.training_set, cd_1.validation_set)

except Exception:
    logger.error(traceback.format_exc())
