"""
Methods for downloading and converting the MNIST dataset to TF-records

implementation is heavily inspired by the slim.datasets implementation (https://github.com/tensorflow/models/tree/master/research/slim/datasets)
"""

import gzip
import os
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf
import src.data.util_data as util_data

# The URLs where the MNIST data can be downloaded.
_DATA_URL = 'http://yann.lecun.com/exdb/mnist/'
_TRAIN_DATA_FILENAME = 'train-images-idx3-ubyte.gz'
_TRAIN_LABELS_FILENAME = 'train-labels-idx1-ubyte.gz'
_TEST_DATA_FILENAME = 't10k-images-idx3-ubyte.gz'
_TEST_LABELS_FILENAME = 't10k-labels-idx1-ubyte.gz'

# Local directories to store the dataset
_DIR_RAW = 'data/raw/MNIST'
_DIR_PROCESSED = 'data/processed/MNIST'

#
_IMAGE_SIZE = 28
_NUM_CHANNELS = 1
_NUM_TRAIN_SAMPLES = 60000
_NUM_TEST_SAMPLES = 10000

# The names of the classes.
_CLASS_NAMES = [
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
]

def _extract_images(filename, num_images):
    """Extract the images into a numpy array.

    Args:
      filename: The path to an MNIST images file.
      num_images: The number of images in the file.

    Returns:
      A numpy array of shape [number_of_images, height, width, channels].
    """
    print('Extracting images from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            _IMAGE_SIZE * _IMAGE_SIZE * num_images * _NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    return data


def _extract_labels(filename, num_labels):
    """Extract the labels into a vector of int64 label IDs.

    Args:
      filename: The path to an MNIST labels file.
      num_labels: The number of labels in the file.

    Returns:
      A numpy array of shape [number_of_labels]
    """
    print('Extracting labels from: ', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def _convert_to_tfrecord(data_filename, labels_filename, num_images, tfrecord_writer):
    """Loads data from the binary MNIST files and writes files to a TFRecord.

    Args:
        data_filename: The filename of the MNIST images.
        labels_filename: The filename of the MNIST labels.
        num_images: The number of images in the dataset.
        tfrecord_writer: The TFRecord writer to use for writing.
    """
    images = _extract_images(data_filename, num_images)
    labels = _extract_labels(labels_filename, num_images)

    shape = (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_CHANNELS)
    image = tf.placeholder(dtype=tf.uint8, shape=shape)
    encoded_image = tf.image.encode_png(image)

    with tf.Session('') as sess:
        for j in range(num_images):
            sys.stdout.write('\r>> Converting image %d/%d' % (j + 1, num_images))
            sys.stdout.flush()

            encoded_img = sess.run(encoded_image, feed_dict={image: images[j]})

            example = util_data.encode_image(
                image_data = encoded_img,
                image_format = 'png'.encode(),
                class_lbl = labels[j],
                class_text = _CLASS_NAMES[labels[j]].encode(),
                height = _IMAGE_SIZE,
                width = _IMAGE_SIZE)

            tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(dataset_dir, split_name):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/%s.tfrecord' % (dataset_dir, split_name)


def download():
  """Downloads MNIST locally.
  """
  for filename in [_TRAIN_DATA_FILENAME,
                   _TRAIN_LABELS_FILENAME,
                   _TEST_DATA_FILENAME,
                   _TEST_LABELS_FILENAME]:
    filepath = os.path.join(_DIR_RAW, filename)

    if not os.path.exists(filepath):
      print('Downloading file %s...' % filename)
      def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %.1f%%' % (
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
      filepath, _ = urllib.request.urlretrieve(_DATA_URL + filename,
                                               filepath,
                                               _progress)
      print()
      with tf.gfile.GFile(filepath) as f:
        size = f.size()
      print('Successfully downloaded', filename, size, 'bytes.')


def process():
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(_DIR_PROCESSED):
        tf.gfile.MakeDirs(_DIR_PROCESSED)

    training_filename = _get_output_filename(_DIR_PROCESSED, 'train')
    testing_filename = _get_output_filename(_DIR_PROCESSED, 'test')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        data_filename = os.path.join(_DIR_RAW, _TRAIN_DATA_FILENAME)
        labels_filename = os.path.join(_DIR_RAW, _TRAIN_LABELS_FILENAME)

        _convert_to_tfrecord(data_filename, labels_filename, _NUM_TRAIN_SAMPLES, tfrecord_writer)

    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        data_filename = os.path.join(_DIR_RAW, _TEST_DATA_FILENAME)
        labels_filename = os.path.join(_DIR_RAW, _TEST_LABELS_FILENAME)

        _convert_to_tfrecord(data_filename, labels_filename, _NUM_TEST_SAMPLES, tfrecord_writer)


    print('\nFinished converting the MNIST dataset!')
