
import argparse
import csv
import datetime
import errno
import io
import logging
import os
import subprocess
import sys

import apache_beam as beam
from apache_beam.metrics import Metrics

try:
  try:
    from apache_beam.options.pipeline_options import PipelineOptions
  except ImportError:
    from apache_beam.utils.pipeline_options import PipelineOptions
except ImportError:
  from apache_beam.utils.options import PipelineOptions

from PIL import Image
import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import inception_v3 as inception
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io

slim = tf.contrib.slim

error_count = Metrics.counter('main', 'errorCount')
missing_label_count = Metrics.counter('main', 'missingLabelCount')
csv_rows_count = Metrics.counter('main', 'csvRowsCount')
labels_count = Metrics.counter('main', 'labelsCount')
labels_without_ids = Metrics.counter('main', 'labelsWithoutIds')
existing_file = Metrics.counter('main', 'existingFile')
non_existing_file = Metrics.counter('main', 'nonExistingFile')
skipped_empty_line = Metrics.counter('main', 'skippedEmptyLine')
embedding_good = Metrics.counter('main', 'embedding_good')
embedding_bad = Metrics.counter('main', 'embedding_bad')
incompatible_image = Metrics.counter('main', 'incompatible_image')
invalid_uri = Metrics.counter('main', 'invalid_file_name')
unlabeled_image = Metrics.counter('main', 'unlabeled_image')
unknown_label = Metrics.counter('main', 'unknown_label')


class Default(object):
  """Default values of variables."""
  FORMAT = 'jpeg'

class ReadAndTransformImageDoFn(beam.DoFn):
  """Read files from GCS and apply transformation on image.

  """

  def process(self, element, resize_shape):
    try:
      uri, label_ids = element.element
    except AttributeError:
      uri, label_ids = element

    # TF will enable 'rb' in future versions, but until then, 'r' is
    # required.
    def _open_file_read_binary(uri):
      try:
        return file_io.FileIO(uri, mode='rb')
      except errors.InvalidArgumentError:
        return file_io.FileIO(uri, mode='r')

    try:
      with _open_file_read_binary(uri) as f:
        image_bytes = f.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('L')

      if isinstance(resize_shape, tuple):
        img = img.resize(resize_shape, resample=Image.BICUBIC)

    # A variety of different calling libraries throw different exceptions here.
    # They all correspond to an unreadable file so we treat them equivalently.
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error processing image %s: %s', uri, str(e))
      error_count.inc()
      return

    # Convert to desired format and output.
    output = io.BytesIO()
    img.save(output, Default.FORMAT)
    image_bytes = output.getvalue()
    yield uri, label_ids, image_bytes


class TFExampleFromImageDoFn(beam.DoFn):
  """Embeds image bytes and labels, stores them in tensorflow.Example.

  (uri, label_id, image_bytes) -> (tensorflow.Example).

  Attributes:
    image_graph_uri: an uri to gcs bucket where serialized image graph is
                     stored.
  """

  def process(self, element):

    def _bytes_feature(value):
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    try:
      element = element.element
    except AttributeError:
      pass
    uri, label_id, image_bytes = element

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_uri': _bytes_feature([uri]),
        'image_bytes': _bytes_feature([image_bytes]),
        'label': _int64_feature([label_id])
    }))

    yield example


def configure_pipeline(p, opt):
  """Specify PCollection and transformations in pipeline."""
  read_input_source = beam.io.ReadFromText(
      opt.input_path, strip_trailing_newlines=True)

  _ = (p
       | 'Read input' >> read_input_source
       | 'Parse input' >> beam.Map(lambda line: csv.reader([line]).next())
       | 'Read and transform image' >> beam.ParDo(ReadAndTransformImageDoFn(), (opt.resize_width, opt.resize_height))
       | 'Make TFExample' >> beam.ParDo(TFExampleFromImageDoFn())
       | 'SerializeToString' >> beam.Map(lambda x: x.SerializeToString())
       | 'Save to disk' >> beam.io.WriteToTFRecord(opt.output_path, file_name_suffix='.tfrecord.gz'))


def run(in_args=None):
  """Runs the pre-processing pipeline."""
  pipeline_options = PipelineOptions.from_dictionary(vars(in_args))
  with beam.Pipeline(options=pipeline_options) as p:
    configure_pipeline(p, in_args)


def default_args(argv):
  """Provides default values for Workflow flags."""
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--input_path',
      required=True,
      help='Input specified as uri to CSV file. Each line of csv file '
      'contains colon-separated GCS uri to an image and label id.')
  parser.add_argument(
      '--output_path',
      required=True,
      help='Output directory to write results to.')
  parser.add_argument(
      '--resize',
      required=True,
      type=int,
      help='Whether to resize the image'
  )
  parser.add_argument(
      '--resize_width',
      required=False,
      type=int,
      default=224,
      help='The resizing image width'
  )
  parser.add_argument(
      '--resize_height',
      required=False,
      type=int,
      default=224,
      help='the resizing image height'
  )

  parser.add_argument(
      '--project',
      type=str,
      help='The cloud project name to be used for running this pipeline')
  parser.add_argument(
      '--job_name',
      type=str,
      default='image-classification-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
      help='A unique job identifier.')
  parser.add_argument(
      '--num_workers', default=20, type=int, help='The number of workers.')
  parser.add_argument('--cloud', default=False, action='store_true')
  parser.add_argument(
      '--runner',
      help='See Dataflow runners, may be blocking'
      ' or not, on cloud or not, etc.')

  parsed_args, _ = parser.parse_known_args(argv)

  if parsed_args.cloud:
    # Flags which need to be set for cloud runs.
    default_values = {
        'project':
            get_cloud_project(),
        'temp_location':
            os.path.join(os.path.dirname(parsed_args.output_path), 'temp'),
        'runner':
            'DataflowRunner',
        'save_main_session':
            True,
    }
  else:
    # Flags which need to be set for local runs.
    default_values = {
        'runner': 'DirectRunner',
    }

  for kk, vv in default_values.iteritems():
    if kk not in parsed_args or not vars(parsed_args)[kk]:
      vars(parsed_args)[kk] = vv

  return parsed_args


def get_cloud_project():
  cmd = [
      'gcloud', '-q', 'config', 'list', 'project',
      '--format=value(core.project)'
  ]
  with open(os.devnull, 'w') as dev_null:
    try:
      res = subprocess.check_output(cmd, stderr=dev_null).strip()
      if not res:
        raise Exception('--cloud specified but no Google Cloud Platform '
                        'project found.\n'
                        'Please specify your project name with the --project '
                        'flag or set a default project: '
                        'gcloud config set project YOUR_PROJECT_NAME')
      return res
    except OSError as e:
      if e.errno == errno.ENOENT:
        raise Exception('gcloud is not installed. The Google Cloud SDK is '
                        'necessary to communicate with the Cloud ML service. '
                        'Please install and set up gcloud.')
      raise


def main(argv):
    try:
        arg_dict = default_args(argv)
        run(arg_dict)

    except Exception as e:

        print("Program error: ", str(e))


if __name__ == '__main__':
  main(sys.argv[1:])
