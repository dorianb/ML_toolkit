
import argparse
import logging

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils

import util
from util import override_if_not_in_args

slim = tf.contrib.slim

LOGITS_TENSOR_NAME = 'logits_tensor'
IMAGE_URI_COLUMN = 'image_uri'
LABEL_COLUMN = 'label'
EMBEDDING_COLUMN = 'embedding'

BOTTLENECK_TENSOR_SIZE = 2048

class GraphMod():
  TRAIN = 1
  EVALUATE = 2
  PREDICT = 3


def build_signature(inputs, outputs):
  """Build the signature.

  Not using predic_signature_def in saved_model because it is replacing the
  tensor name, b/35900497.

  Args:
    inputs: a dictionary of tensor name to tensor
    outputs: a dictionary of tensor name to tensor
  Returns:
    The signature, a SignatureDef proto.
  """
  signature_inputs = {key: saved_model_utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}
  signature_outputs = {key: saved_model_utils.build_tensor_info(tensor)
                       for key, tensor in outputs.items()}

  signature_def = signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)

  return signature_def


def create_model():
  """Factory method that creates model to be used by generic task.py."""
  parser = argparse.ArgumentParser()
  # Label count needs to correspond to nubmer of labels in dictionary used
  # during preprocessing.
  parser.add_argument('--label_count', type=int, default=5)
  parser.add_argument('--dropout', type=float, default=0.5)
  parser.add_argument('--checkpoint_file', type=str, default='')
  args, task_args = parser.parse_known_args()
  override_if_not_in_args('--max_steps', '1000', task_args)
  override_if_not_in_args('--batch_size', '100', task_args)
  override_if_not_in_args('--eval_set_size', '370', task_args)
  override_if_not_in_args('--eval_interval_secs', '2', task_args)
  override_if_not_in_args('--log_interval_secs', '2', task_args)
  override_if_not_in_args('--min_train_eval_rate', '2', task_args)
  return Model(args.label_count, args.dropout,
               args.checkpoint_file), task_args


class GraphReferences(object):
  """Holder of base tensors used for training model using common task."""

  def __init__(self):
    self.examples = None
    self.train = None
    self.global_step = None
    self.metric_updates = []
    self.metric_values = []
    self.keys = None
    self.predictions = []
    self.input_jpeg = None


class Model(object):
  """TensorFlow model for the flowers problem."""

  def __init__(self, label_count, dropout, inception_checkpoint_file):
    self.label_count = label_count
    self.dropout = dropout
    self.inception_checkpoint_file = inception_checkpoint_file

  def build_graph(self, data_paths, batch_size, graph_mod):
    """Builds generic graph for training or eval."""
    tensors = GraphReferences()
    is_training = graph_mod == GraphMod.TRAIN
    if data_paths:
      tensors.keys, tensors.examples = util.read_examples(
          data_paths,
          batch_size,
          shuffle=is_training,
          num_epochs=None if is_training else 2)
    else:
      tensors.examples = tf.placeholder(tf.string, name='input', shape=(None,))

    if graph_mod == GraphMod.PREDICT:
      pass
    else:
      # For training and evaluation we assume data is preprocessed, so the
      # inputs are tf-examples.
      # Generate placeholders for examples.
      with tf.name_scope('inputs'):
        feature_map = {
            'image_uri':
                tf.FixedLenFeature(
                    shape=[], dtype=tf.string, default_value=['']),
            'image_bytes':
                tf.FixedLenFeature(
                    shape=[], dtype=tf.string, default_value=['']),
            'label':
                tf.FixedLenFeature(
                    shape=[1], dtype=tf.int64,
                    default_value=[self.label_count])
        }
        parsed = tf.parse_example(tensors.examples, features=feature_map)
        labels = tf.squeeze(parsed['label'])
        uris = tf.squeeze(parsed['image_uri'])
        images_str_tensor = parsed['image_bytes']

      def decode_and_resize(image_str_tensor):
          """Decodes jpeg string, resizes it and returns a uint8 tensor."""
          image = tf.image.decode_jpeg(image_str_tensor, channels=1)

          # Note resize expects a batch_size, but tf_map supresses that index,
          # thus we have to expand then squeeze.  Resize returns float32 in the
          # range [0, uint8_max]
          """
          image = tf.expand_dims(image, 0)
          image = tf.image.resize_bilinear(
              image, [height, width], align_corners=False)*/
          image = tf.squeeze(image, squeeze_dims=[0])
          """

          image = tf.cast(image, dtype=tf.uint8)

          # convert_image_dtype, also scales [0, uint8_max] -> [0 ,1).
          return tf.image.convert_image_dtype(image, dtype=tf.float32)

    images = tf.map_fn(
      decode_and_resize, images_str_tensor, back_prop=False, dtype=tf.float32)

    # We assume a default label, so the total number of labels is equal to
    # label_count+1.
    all_labels_count = self.label_count + 1

    with tf.name_scope('model'):

        fc_padding = 'VALID'
        with tf.variable_scope('model', 'vgg_16', [images]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=fc_padding, scope='fc6')
                net = slim.dropout(net, 0.5, is_training=True,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)

                net = slim.dropout(net, 0.5, is_training=True,
                                   scope='dropout7')
                net = slim.conv2d(net, all_labels_count, [1, 1],
                                  activation_fn=None,
                                  normalizer_fn=None,
                                  scope='fc8')

                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
    logits = net
    softmax = tf.nn.softmax(logits)
    # Prediction is the index of the label with the highest score. We are
    # interested only in the top score.
    prediction = tf.argmax(softmax, 1)
    tensors.predictions = [prediction, softmax, images]

    if graph_mod == GraphMod.PREDICT:
      return tensors

    with tf.name_scope('evaluate'):
      loss_value = loss(logits, labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    if is_training:
      tensors.train, tensors.global_step = training(loss_value)
    else:
      tensors.global_step = tf.Variable(0, name='global_step', trainable=False)

    # Add means across all batches.
    loss_updates, loss_op = util.loss(loss_value)
    accuracy_updates, accuracy_op = util.accuracy(logits, labels)

    if not is_training:
      tf.summary.scalar('accuracy', accuracy_op)
      tf.summary.scalar('loss', loss_op)

    tensors.metric_updates = loss_updates + accuracy_updates
    tensors.metric_values = [loss_op, accuracy_op]
    return tensors

  def build_train_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.TRAIN)

  def build_eval_graph(self, data_paths, batch_size):
    return self.build_graph(data_paths, batch_size, GraphMod.EVALUATE)

  def restore_from_checkpoint(self, session, inception_checkpoint_file,
                              trained_checkpoint_file):
    """To restore model variables from the checkpoint file.

       The graph is assumed to consist of an inception model and other
       layers including a softmax and a fully connected layer. The former is
       pre-trained and the latter is trained using the pre-processed data. So
       we restore this from two checkpoint files.
    Args:
      session: The session to be used for restoring from checkpoint.
      inception_checkpoint_file: Path to the checkpoint file for the Inception
                                 graph.
      trained_checkpoint_file: path to the trained checkpoint for the other
                               layers.
    """
    inception_exclude_scopes = [
        'InceptionV3/AuxLogits', 'InceptionV3/Logits', 'global_step',
        'final_ops'
    ]
    reader = tf.train.NewCheckpointReader(inception_checkpoint_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Get all variables to restore. Exclude Logits and AuxLogits because they
    # depend on the input data and we do not need to intialize them.
    all_vars = tf.contrib.slim.get_variables_to_restore(
        exclude=inception_exclude_scopes)
    # Remove variables that do not exist in the inception checkpoint (for
    # example the final softmax and fully-connected layers).
    inception_vars = {
        var.op.name: var
        for var in all_vars if var.op.name in var_to_shape_map
    }
    inception_saver = tf.train.Saver(inception_vars)
    inception_saver.restore(session, inception_checkpoint_file)

    # Restore the rest of the variables from the trained checkpoint.
    trained_vars = tf.contrib.slim.get_variables_to_restore(
        exclude=inception_exclude_scopes + inception_vars.keys())
    trained_saver = tf.train.Saver(trained_vars)
    trained_saver.restore(session, trained_checkpoint_file)

  def build_prediction_graph(self):
    """Builds prediction graph and registers appropriate endpoints."""

    tensors = self.build_graph(None, 1, GraphMod.PREDICT)

    keys_placeholder = tf.placeholder(tf.string, shape=[None])
    inputs = {
        'key': keys_placeholder,
        'image_bytes': tensors.input_jpeg
    }

    # To extract the id, we need to add the identity function.
    keys = tf.identity(keys_placeholder)
    outputs = {
        'key': keys,
        'prediction': tensors.predictions[0],
        'scores': tensors.predictions[1]
    }

    return inputs, outputs

  def export(self, last_checkpoint, output_dir):
    """Builds a prediction graph and exports the model.

    Args:
      last_checkpoint: Path to the latest checkpoint file from training.
      output_dir: Path to the folder to be used to output the model.
    """
    logging.info('Exporting prediction graph to %s', output_dir)
    with tf.Session(graph=tf.Graph()) as sess:
      # Build and save prediction meta graph and trained variable values.
      inputs, outputs = self.build_prediction_graph()
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      self.restore_from_checkpoint(sess, self.inception_checkpoint_file,
                                   last_checkpoint)
      signature_def = build_signature(inputs=inputs, outputs=outputs)
      signature_def_map = {
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
      }
      builder = saved_model_builder.SavedModelBuilder(output_dir)
      builder.add_meta_graph_and_variables(
          sess,
          tags=[tag_constants.SERVING],
          signature_def_map=signature_def_map)
      builder.save()

  def format_metric_values(self, metric_values):
    """Formats metric values - used for logging purpose."""

    # Early in training, metric_values may actually be None.
    loss_str = 'N/A'
    accuracy_str = 'N/A'
    try:
      loss_str = '%.3f' % metric_values[0]
      accuracy_str = '%.3f' % metric_values[1]
    except (TypeError, IndexError):
      pass

    return '%s, %s' % (loss_str, accuracy_str)


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss_op):
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(epsilon=0.001)
    train_op = optimizer.minimize(loss_op, global_step)
    return train_op, global_step
