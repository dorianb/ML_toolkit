import argparse
import logging
<<<<<<< HEAD
import os
import traceback
import json
import pandas as pd
import numpy as np

from sequence_model.classRNN import RNN
from sequence_model.classLSTM import LSTM

parser = argparse.ArgumentParser(description='Sequence program')
parser.add_argument('--dataset-path', type=str, help='Path to the dataset', default=".")
parser.add_argument('--metadata-path', type=str, help='Path to the metadata', default=".")
parser.add_argument('--model', type=str, help='Name of the model', default="rnn")
parser.add_argument('--units', type=str, help='List of units for each cells', default="[[100, 100]]")
parser.add_argument('--time-steps', type=int, help='Sequence length', default=24)
parser.add_argument('--n-features', type=int, help='Number of features', default=1)
parser.add_argument('--n-output', type=int, help='Number of output', default=1)
parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
parser.add_argument('--train-size', type=float, help='Training set size', default=0.7)
parser.add_argument('--validation-size', type=float, help='Validation set size', default=0.2)
parser.add_argument('--test-size', type=float, help='Test set size', default=0.1)
parser.add_argument('--train', type=int, help='Training mode', default=1)
parser.add_argument('--optimizer', type=str, help='Optimizer', default='adam')
parser.add_argument('--learning-rate', type=float, help='Learning rate', default=0.01)
parser.add_argument('--from-pretrained', type=int, help='Transfer learning mode', default=0)
parser.add_argument('--name', type=str, help='The unique name of the program', default="")
parser.add_argument('--debug', type=int, help='Debug mode', default=0)
args = parser.parse_args()

logger = logging.Logger("sequence_exec", level=logging.DEBUG if args.debug else logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.DEBUG if args.debug else logging.INFO)
logger.addHandler(consoleHandler)

fileHandler = logging.FileHandler("sequence_exec.log")
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(fileHandler)

assert os.path.isdir(args.dataset_path), "{0} is not a valid directory".format(args.dataset_path)


class UnknwonDataSetKind(Exception):
    pass


def load_dataset(kind="words", filename="dinos.txt", seq_length=24):
    """
    Load the dataset

    Args:
        kind:
        filename:

    Returns:
        training and test set
    """
    if kind == "words":

        words = pd.read_csv(filename, header=None)
        characters = list(set(np.char.lower(words.values.ravel().sum()).tolist())) + ['\n']
        n_values = len(characters)

        dataset = np.concatenate([
            np.concatenate([
                np.eye(n_values)[characters.index(np.char.lower(row[0]).tolist()[i])]
                if i < len(np.char.lower(row[0]).tolist()) else np.eye(n_values)[-1]
                for i in range(seq_length)
            ], axis=0).reshape((seq_length, n_values))
            for index, row in words.iterrows()
        ], axis=0).reshape((len(words), seq_length, n_values))

        np.random.shuffle(dataset)

        return dataset[:-10], dataset[-10:]

    else:
        raise UnknwonDataSetKind()

try:

    training_set, validation_set = load_dataset(kind=args.kind, filename=args.filename, seq_length=args.time_steps)

    if args.model == "lstm":
        lstm_1 = LSTM()
        lstm_1.fit(training_set, validation_set)

    else:
        rnn_1 = RNN(json.load(args.units), args.f_out, batch_size=args.batch_size, time_steps=args.time_steps,
                    n_features=args.n_features, n_output=args.n_output, with_prev_output=args.with_prev_output,
                    with_input=args.with_input, return_sequences=args.return_sequences,
                    n_epochs=args.epochs, validation_step=10, checkpoint_step=100,
                    from_pretrained=False, optimizer_name=args.optimizer,
                    learning_rate=args.learning_rate, loss_name=args.loss,
                    summary_path=os.path.join(args.metadata_path, "summaries"),
                    checkpoint_path=os.path.join(args.metadata_path, "checkpoints"),
                    name=args.name, logger=logger, debug=args.debug)

        rnn_1.fit(training_set, validation_set)

except Exception:
    logger.error(traceback.format_exc())
=======

from rnn.classLSTM import LSTM

"""
src/model/rnn/exec/lstm_train_predict_exec.py \
    --train 1 \
    --debug 1 
"""

parser = argparse.ArgumentParser(description='LSTM Train and predict')
parser.add_argument('--train', type=int, help='Training mode', default=1)
parser.add_argument('--', type=int, help='Training mode', default=1)
parser.add_argument('--debug', type=int, help='Debug mode', default=0)
args = parser.parse_args()

logger = logging.Logger("lstm_exec", level=logging.DEBUG if args.debug else logging.INFO)

try:

    lstm_0 = LSTM(batch_size=1, time_step=3, n_features=3)

except Exception:

    logger.error('Program exited with error')
>>>>>>> 1db20179ac4cb8b4a03f314a8091bfd6847c320d
