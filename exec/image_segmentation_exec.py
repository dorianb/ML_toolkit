import argparse
import logging
import os
import traceback

from computer_vision.classWNet import WNet

parser = argparse.ArgumentParser(description='Image segmentation program')
parser.add_argument('--dataset-path', type=str, help='Path to the dataset', default=".")
parser.add_argument('--metadata-path', type=str, help='Path to the metadata', default=".")
parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
parser.add_argument('--train', type=int, help='Training mode', default=1)
parser.add_argument('--n-epoch', type=int, help='Number of epochs', default=1)
parser.add_argument('--n-class', type=int, help='Number of classes', default=2)
parser.add_argument('--optimizer', type=str, help='Optimizer', default='adam')
parser.add_argument('--initializer', type=str, help='Initializer', default='zeros')
parser.add_argument('--padding', type=str, help='Padding', default='SAME')
parser.add_argument('--learning-rate', type=float, help='Learning rate', default=0.01)
parser.add_argument('--from-pretrained', type=int, help='Transfer learning mode', default=0)
parser.add_argument('--checkpoint-step', type=int, help='Number of iterations between two checkpoints', default=100)
parser.add_argument('--grayscale', type=int, help='Whether to load image as grayscale', default=0)
parser.add_argument('--rgb', type=int, help='Whether to load image encoded as rgb', default=1)
parser.add_argument('--binarize', type=int, help='Whether to binarize image', default=0)
parser.add_argument('--normalize', type=int, help='Whether to normalize image dimensions', default=0)
parser.add_argument('--name', type=str, help='The unique name of the program', default="wnet")
parser.add_argument('--debug', type=int, help='Debug mode', default=0)
args = parser.parse_args()

logger = logging.Logger("image_segmentation_exec",
                        level=logging.DEBUG if args.debug else logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG if args.debug else logging.INFO)
logger.addHandler(consoleHandler)

fileHandler = logging.FileHandler("image_segmentation_exec.log")
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)

assert os.path.isdir(args.dataset_path), "{0} is not a valid directory".format(args.dataset_path)

try:

    training_set = [os.path.join(args.dataset_path, filename) for filename in os.listdir(args.dataset_path)]
    logger.info("%d images in training set" % len(training_set))

    resize_dim = (224, 224) if args.batch_size > 1 else None

    wnet_1 = WNet(
        batch_size=args.batch_size, n_channel=3, initializer_name=args.initializer,
        padding=args.padding, k=args.n_class, from_pretrained=args.from_pretrained,
        optimizer_name=args.optimizer, learning_rate=args.learning_rate,
        n_epochs=args.n_epoch, checkpoint_step=args.checkpoint_step,
        grayscale=args.grayscale, rgb=args.rgb, binarize=args.binarize, normalize=args.normalize,
        resize_dim=resize_dim, metadata_path=args.metadata_path, logger=logger, name=args.name,
        debug=args.debug)

    wnet_1.fit(training_set, [])

except Exception:
    logger.error(traceback.format_exc())
