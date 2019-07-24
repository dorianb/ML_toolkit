import argparse
import logging
import os
import traceback

from computer_vision.classResNet import ResNet
from dataset_utils.classImageClassificationDataset import ImageClassificationDataset

parser = argparse.ArgumentParser(description='Image classification program')
parser.add_argument('--n-layers', type=int, help='Number of layers (18, 34, 50, 101, 152)', default=18)
parser.add_argument('--training-set-path', type=str, help='Path to the training set', default=".")
parser.add_argument('--test-set-path', type=str, help='Path to the test set', default=".")
parser.add_argument('--prediction-path', type=str, help='Path to the predictions to be save', default=".")
parser.add_argument('--metadata-path', type=str, help='Path to the metadata', default=".")
parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
parser.add_argument('--train-size', type=float, help='Training set size', default=0.7)
parser.add_argument('--validation-size', type=float, help='Validation set size', default=0.2)
parser.add_argument('--test-size', type=float, help='Test set size', default=0.1)
parser.add_argument('--mode', type=str, help='Mode (train, predict)', default=1)
parser.add_argument('--optimizer', type=str, help='Optimizer', default='adam')
parser.add_argument('--learning-rate', type=float, help='Learning rate', default=0.01)
parser.add_argument('--n-epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--height', type=int, help='Height of image', default=1)
parser.add_argument('--width', type=int, help='Width of image', default=1)
parser.add_argument('--from-pretrained', type=int, help='Transfer learning mode', default=0)
parser.add_argument('--name', type=str, help='The unique name of the program', default="vgg")
parser.add_argument('--debug', type=int, help='Debug mode', default=0)
args = parser.parse_args()

logger = logging.Logger("image_classification_resnet_exec",
                        level=logging.DEBUG if args.debug else logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG if args.debug else logging.INFO)
logger.addHandler(consoleHandler)

fileHandler = logging.FileHandler("image_classification_resnet_exec.log")
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)

assert os.path.isdir(args.training_set_path), "{0} is not a valid directory".format(args.training_set_path)
assert os.path.isdir(args.test_set_path), "{0} is not a valid directory".format(args.test_set_path)
assert os.path.isdir(args.prediction_path), "{0} is not a valid directory".format(args.prediction_path)
assert os.path.isdir(args.metadata_path), "{0} is not a valid directory".format(args.metadata_path)


try:

    cd_1 = ImageClassificationDataset(
        args.training_set_path, testing_path=args.test_set_path,
        train_size=args.train_size, val_size=args.validation_size, test_size=args.test_size)
    classes = cd_1.labels

    if args.mode == "train":

        resnet_1 = ResNet(classes, n_layers=args.n_layers, batch_size=args.batch_size,
                    height=args.height, width=args.width,
                    dim_out=len(classes), grayscale=True, binarize=False, normalize=False,
                    learning_rate=args.learning_rate, n_epochs=args.n_epochs, validation_step=100,
                    checkpoint_step=100, is_encoder=False, validation_size=300,
                    optimizer=args.optimizer, metadata_path=args.metadata_path,
                    name=args.name, from_pretrained=args.from_pretrained, is_training=True,
                    logger=logger, debug=args.debug)

        resnet_1.fit(cd_1.training_set, cd_1.validation_set)

    elif args.mode == "predict":

        resnet_1 = ResNet(classes, n_layers=args.n_layers,
                    batch_size=args.batch_size, height=args.height, width=args.width,
                    dim_out=len(classes), grayscale=True, binarize=False, normalize=False,
                    metadata_path=args.metadata_path,
                    name=args.name, from_pretrained=True, is_training=False,
                    logger=logger, debug=args.debug)

        predictions = resnet_1.predict(cd_1.testing_examples)
        cd_1.save_test_pred(predictions, args.prediction_path)

except Exception:
    logger.error(traceback.format_exc())
