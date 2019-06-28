import argparse
import pandas as pd
import numpy as np
import os
import traceback

from keras import Input, Model, metrics, optimizers, activations
from keras.layers import SimpleRNNCell, LSTMCell, LSTM, SimpleRNN, Dense, TimeDistributed, Masking

from sequence_model.classRNNMultiLayer import RNNMultiLayer
from sequence_model.classLSTMMultiLayer import LSTMMultiLayer

parser = argparse.ArgumentParser(description='Sequential model')
parser.add_argument('--model', type=str, help='Model name', default="RNN")
parser.add_argument('--target', type=str, help='Column name target', default="nb_product_size_sold_at_sale_end")
parser.add_argument('--dataset-prepared', type=int, help='Whether to load a prepared dataset', default=0)
parser.add_argument('--dataset-path', type=str, help='Path to the dataset', default=".")
parser.add_argument('--trainset-filename', type=str, help='The filename of the training set', default="train_set.csv")
parser.add_argument('--testset-filename', type=str, help='The filename of the test set', default="test_set.csv")
parser.add_argument('--features-filename', type=str, help='The filename of the features set', default="features.csv")
parser.add_argument('--metadata-path', type=str, help='Path to the metadata', default=".")
parser.add_argument('--units', type=int, help='Number of units per cell', default=100)
parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
parser.add_argument('--time-steps', type=int, help='the length of sequence', default=24)
parser.add_argument('--train', type=int, help='Training mode', default=1)
parser.add_argument('--f-out', type=str, help='Activation function for output', default="identity")
parser.add_argument('--optimizer', type=str, help='Optimizer', default='rmsprop')
parser.add_argument('--learning-rate', type=float, help='Learning rate', default=0.01)
parser.add_argument('--loss-name', type=str, help='Loss rate', default="mse")
parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--layers', type=int, help='Number of layers', default=1)
parser.add_argument('--validation-step', type=int, help='Batch step for evaluation', default=10)
parser.add_argument('--checkpoint-step', type=int, help='Batch step for checkpoint', default=500)
parser.add_argument('--from-pretrained', type=int, help='Transfer learning mode', default=0)
parser.add_argument('--normalize-input', type=int, help='Whether to normalize the input features', default=0)
parser.add_argument('--stop-at-step', type=int, help='Step from which to stop training (0 otherwise)', default=0)
parser.add_argument('--name', type=str, help='The unique name of the program', default="rnn")
parser.add_argument('--debug', type=int, help='Debug mode', default=0)
args = parser.parse_args()

assert os.path.isdir(args.dataset_path), "{0} is not a valid directory".format(args.dataset_path)
assert os.path.isdir(args.metadata_path), "{0} is not a valid directory".format(args.metadata_path)


def normalize_dataset(train_set, test_set):
    """
    Normalize the dataset.

    Args:
        train_set: the training set used for fitting the normalizer and to be transform
        test_set: the test set to be transform

    Returns:
        tuple of train an test set normalized

    """
    train_set_stacked = np.stack(train_set[:, 0])
    test_set_stacked = np.stack(test_set[:, 0])

    f_min = train_set_stacked.min(axis=0).min(axis=0)
    f_max = train_set_stacked.max(axis=0).max(axis=0)

    X_std = (train_set_stacked - f_min) / (f_max - f_min)
    train_set[:, 0] = [x for x in X_std]

    X_std = (test_set_stacked - f_min) / (f_max - f_min)
    test_set[:, 0] = [x for x in X_std]

    return train_set, test_set


def load_dataset(dataset_path, features_path, seq_length=24, shuffle=True, target="nb_product_size_sold_at_sale_end"):
    """
    Load the dataset

    Args:
        dataset_path: the path to the dataset
        features_path: the path to the features name file
        seq_length: the sequence length of input
        shuffle: whether to shuffle dataset
        target: the column used as target
    Returns:
        a tuple of numpy arrays representing the training and validation set
    """
    set = pd.read_csv(dataset_path)
    set_index = set[["language_id", "product_id", "style_value_id", "hour_of_sale"]].drop_duplicates()
    features = pd.read_csv(features_path, header=None).values[:, 0].tolist()
    dataset = []

    for index, row in set_index.iterrows():

        val_index = (
                (set["language_id"] == row["language_id"]) &
                (set["product_id"] == row["product_id"]) &
                (set["style_value_id"] == row["style_value_id"]) &
                (set["hour_of_sale"] <= row["hour_of_sale"]) &
                (set["hour_of_sale"] > row["hour_of_sale"] - seq_length)
        )

        example = set.loc[val_index, features].values
        time_steps = example.shape[0]

        if time_steps < seq_length:
            index = np.arange(seq_length - time_steps, seq_length)
            new_example = np.zeros((seq_length, len(features)))
            new_example[index, :] = example
            example = new_example

        label = set.loc[val_index, target].values[0:1].reshape(1)
        dataset.append([example, label])

    dataset = np.stack(dataset)

    if shuffle:
        np.random.shuffle(dataset)

    return dataset


def get_optimizer(name, lr):
    """

    Args:
        name:
        lr:

    Returns:

    """
    if name.lower() == "adam":
        return optimizers.adam(lr=lr, clipvalue=1.0)
    elif name.lower() == "rmsprop":
        return optimizers.rmsprop(lr=lr, clipvalue=1.0)
    elif name.lower() == "adadelta":
        return optimizers.adadelta(lr=lr, clipvalue=1.0)
    else:
        return optimizers.adam(lr=lr, clipvalue=1.0)


def get_activation(name):
    """

    Args:
        name:

    Returns:

    """
    if name == "identity":
        return activations.linear
    else:
        return name


def split_input_label(examples):
    """
    Load the batch examples.

    Args:
        examples: the example in the batch
    Returns:
        the batch examples
    """
    inputs = []
    labels = []
    for example in examples:
        input, label = example
        input = input.reshape(1, -1) if len(input.shape) == 1 else input
        inputs.append(input)
        labels.append(label)

    return np.stack(inputs), np.stack(labels)

try:

    if args.dataset_prepared:

        training_set = np.load(os.path.join(args.dataset_path, args.trainset_filename))
        test_set = np.load(os.path.join(args.dataset_path, args.testset_filename))

    else:

        training_set = load_dataset(
            os.path.join(args.dataset_path, args.trainset_filename),
            os.path.join(args.dataset_path, args.features_filename),
            seq_length=args.time_steps, target=args.target
        )

        test_set = load_dataset(
            os.path.join(args.dataset_path, args.testset_filename),
            os.path.join(args.dataset_path, args.features_filename),
            seq_length=args.time_steps, shuffle=False, target=args.target
        )

    if args.normalize_input:
        training_set, test_set = normalize_dataset(training_set, test_set)

    n_features = training_set[0][0].shape[-1]

    if args.model.lower() == "lstm":

        model = LSTMMultiLayer(args.units, args.batch_size, args.time_steps, n_features, n_layers=args.layers,
                               is_sequence_output=False, n_output=1, f_out=args.f_out, optimizer_name=args.optimizer,
                               learning_rate=args.learning_rate, loss_name=args.loss_name,
                               epochs=args.epochs, from_pretrained=args.from_pretrained,
                               validation_step=args.validation_step, checkpoint_step=args.checkpoint_step,
                               summary_path=os.path.join(args.metadata_path, "summaries", args.name),
                               checkpoint_path=os.path.join(args.metadata_path, "checkpoints", args.name),
                               name=args.name, debug=args.debug)

    elif args.model.lower() == "lstm_keras":

        x = Input((args.time_steps, n_features))
        X = Masking(mask_value=0, input_shape=(args.time_steps, n_features))(x)
        for l in range(args.layers):
            if l < args.layers - 1:
                X, c, o = LSTM(args.units, return_state=True, return_sequences=True)(X)
            else:
                output = LSTM(args.units, return_state=False, return_sequences=False)(X)

        y = Dense(1, activation=get_activation(args.f_out))(output)

        model = Model(inputs=x, outputs=y)
        optimizer = get_optimizer(args.optimizer, args.learning_rate)
        model.compile(optimizer, loss=args.loss_name, metrics=[metrics.mae, metrics.mse, metrics.mape])
        model.summary()

    elif args.model.lower() == "rnn_keras":

        x = Input((args.time_steps, n_features))
        X = Masking(mask_value=0, input_shape=(args.time_steps, n_features))(x)
        for l in range(args.layers):
            if l < args.layers - 1:
                X, _ = SimpleRNN(args.units, return_state=True, return_sequences=True)(X)
            else:
                output = SimpleRNN(args.units, return_state=False, return_sequences=False)(X)

        y = Dense(1, activation=get_activation(args.f_out))(output)

        model = Model(inputs=x, outputs=y)
        optimizer = get_optimizer(args.optimizer, args.learning_rate)
        model.compile(optimizer, loss=args.loss_name, metrics=[metrics.mae, metrics.mse, metrics.mape])

        model.summary()

    elif args.model.lower() == "lstm_keras_seq":

        x = Input((args.time_steps, n_features))
        X = Masking(mask_value=0, input_shape=(args.time_steps, n_features))(x)
        for l in range(args.layers):
            if l < args.layers - 1:
                X, c, o = LSTM(args.units, return_state=True, return_sequences=True)(X)
            else:
                output = LSTM(args.units, return_state=False, return_sequences=True)(X)

        y = TimeDistributed(Dense(1, activation=get_activation(args.f_out)))(output)

        model = Model(inputs=x, outputs=y)
        optimizer = get_optimizer(args.optimizer, args.learning_rate)
        model.compile(optimizer, loss=args.loss_name, metrics=[metrics.mae, metrics.mse, metrics.mape])

        model.summary()

    elif args.model.lower() == "rnn_keras_seq":

        x = Input((args.time_steps, n_features))
        X = Masking(mask_value=0, input_shape=(args.time_steps, n_features))(x)
        for l in range(args.layers):
            if l < args.layers - 1:
                X, _ = SimpleRNN(args.units, return_state=True, return_sequences=True)(X)
            else:
                output = SimpleRNN(args.units, return_state=False, return_sequences=True)(X)

        y = TimeDistributed(Dense(1, activation=get_activation(args.f_out)))(output)

        model = Model(inputs=x, outputs=y)
        optimizer = get_optimizer(args.optimizer, args.learning_rate)
        model.compile(optimizer, loss=args.loss_name, metrics=[metrics.mae, metrics.mse, metrics.mape])

        model.summary()

    else:

        model = RNNMultiLayer(args.units, args.batch_size, args.time_steps, n_features, n_layers=args.layers,
                              is_sequence_output=False, with_prev_output=False, n_output=1, f_out=args.f_out,
                              optimizer_name=args.optimizer, learning_rate=args.learning_rate,
                              loss_name=args.loss_name, epochs=args.epochs,
                              from_pretrained=args.from_pretrained, validation_step=args.validation_step,
                              checkpoint_step=args.checkpoint_step,
                              summary_path=os.path.join(args.metadata_path, "summaries", args.name),
                              checkpoint_path=os.path.join(args.metadata_path, "checkpoints", args.name),
                              name=args.name, debug=args.debug)

    if args.train:

        if args.model.lower() == "rnn_keras" or args.model.lower() == "lstm_keras":

            x_train, y_train = split_input_label(training_set)
            x_test, y_test = split_input_label(test_set)
            model.fit(x=x_train, y=y_train, batch_size=None, epochs=args.epochs, validation_split=0.0,
                      validation_data=(x_test, y_test), shuffle=True, class_weight=None, sample_weight=None,
                      initial_epoch=0, steps_per_epoch=args.validation_step, validation_steps=args.validation_step)
            model_json = model.to_json()
            with open(args.model.lower() + "_model.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights(args.model.lower() + "_model.h5")

        elif args.model.lower() == "rnn_keras_seq" or args.model.lower() == "lstm_keras_seq":

            x_train, y_train = split_input_label(training_set)
            y_train = np.repeat(y_train, args.time_steps).reshape((len(y_train), args.time_steps, 1))
            x_test, y_test = split_input_label(test_set)
            y_test = np.repeat(y_test, args.time_steps).reshape((len(y_test), args.time_steps, 1))
            model.fit(x=x_train, y=y_train, batch_size=None, epochs=args.epochs, validation_split=0.0,
                      validation_data=(x_test, y_test), shuffle=True, class_weight=None, sample_weight=None,
                      initial_epoch=0, steps_per_epoch=args.validation_step, validation_steps=args.validation_step)
            model_json = model.to_json()
            with open(args.model.lower() + "_model.json", "w") as json_file:
                json_file.write(model_json)
            model.save_weights(args.model.lower() + "_model.h5")

        else:

            model.fit(training_set, test_set, stop_at_step=args.stop_at_step if args.stop_at_step > 0 else None)

        print("Fitted model without error")

    else:
        if args.model.lower() == "rnn_keras_seq" or args.model.lower() == "lstm_keras_seq" or args.model.lower() == "rnn_keras" or args.model.lower() == "lstm_keras":
            model.load_weights(args.model.lower() + "_model.h5")
            x_test, y_test = split_input_label(test_set)
            y_pred = model.predict(x_test, verbose=0)
            np.save(os.path.join(args.dataset_path, "predictions.npy"), y_pred)
        else:
            predictions = model.predict(test_set)
            np.save(os.path.join(args.dataset_path, "predictions.npy"), predictions)

        print("Predicted without error")

except Exception:

    print(traceback.format_exc())
