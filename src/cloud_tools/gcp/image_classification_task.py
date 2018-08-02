import argparse
import os

import computer_vision as model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam


def train(hparams):
  """Run the training and evaluate using the high level API"""

  train_input = lambda: model.input_fn(
      hparams.train_files,
      num_epochs=hparams.num_epochs,
      batch_size=hparams.train_batch_size
  )

  # Don't shuffle evaluation data
  eval_input = lambda: model.input_fn(
      hparams.eval_files,
      batch_size=hparams.eval_batch_size,
      shuffle=False
  )

  train_spec = tf.estimator.TrainSpec(train_input,
                                      max_steps=hparams.train_steps
                                      )

  exporter = tf.estimator.FinalExporter('census',
          model.SERVING_FUNCTIONS[hparams.export_format])
  eval_spec = tf.estimator.EvalSpec(eval_input,
                                    steps=hparams.eval_steps,
                                    exporters=[exporter],
                                    name='census-eval'
                                    )

  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=hparams.job_dir)
  print('model dir {}'.format(run_config.model_dir))
  estimator = model.build_estimator(
      embedding_size=hparams.embedding_size,
      # Construct layers sizes with exponetial decay
      hidden_units=[
          max(2, int(hparams.first_layer_size *
                     hparams.scale_factor**i))
          for i in range(hparams.num_layers)
      ],
      config=run_config
  )

  tf.estimator.train_and_evaluate(estimator,
                                  train_spec,
                                  eval_spec)

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--dataset-path', type=str, help='Path to the dataset', default=".")
  parser.add_argument('--batch-size', type=int, help='Batch size', default=1)
  parser.add_argument('--train-size', type=float, help='Training set size', default=0.7)
  parser.add_argument('--validation-size', type=float, help='Validation set size', default=0.2)
  parser.add_argument('--test-size', type=float, help='Test set size', default=0.1)
  parser.add_argument('--train', type=int, help='Training mode', default=1)
  parser.add_argument('--optimizer', type=str, help='Optimizer', default='adam')
  parser.add_argument('--learning-rate', type=float, help='Learning rate', default=0.01)
  parser.add_argument('--debug', type=int, help='Debug mode', default=0)

  # Input Arguments
  parser.add_argument(
      '--dataset-path',
      help='GCS or local paths to training data',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--num-epochs',
      help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
      type=int,
  )
  parser.add_argument(
      '--train-batch-size',
      help='Batch size for training steps',
      type=int,
      default=40
  )
  parser.add_argument(
      '--eval-batch-size',
      help='Batch size for evaluation steps',
      type=int,
      default=40
  )
  parser.add_argument(
      '--eval-files',
      help='GCS or local paths to evaluation data',
      nargs='+',
      required=True
  )
  # Training arguments
  parser.add_argument(
      '--embedding-size',
      help='Number of embedding dimensions for categorical columns',
      default=8,
      type=int
  )
  parser.add_argument(
      '--first-layer-size',
      help='Number of nodes in the first layer of the DNN',
      default=100,
      type=int
  )
  parser.add_argument(
      '--num-layers',
      help='Number of layers in the DNN',
      default=4,
      type=int
  )
  parser.add_argument(
      '--scale-factor',
      help='How quickly should the size of the layers in the DNN decay',
      default=0.7,
      type=float
  )
  parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
  )

  # Argument to turn on all logging
  parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default='INFO',
  )
  # Experiment arguments
  parser.add_argument(
      '--train-steps',
      help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      help='Number of steps to run evaluation for at each checkpoint',
      default=100,
      type=int
  )
  parser.add_argument(
      '--export-format',
      help='The input format of the exported SavedModel binary',
      choices=['JSON', 'CSV', 'EXAMPLE'],
      default='JSON'
  )

  args = parser.parse_args()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  # Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Run the training job
  hparams=hparam.HParams(**args.__dict__)
  train(hparams)
