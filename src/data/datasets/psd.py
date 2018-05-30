"""
Methods for downloading and converting the MNIST dataset to TF-records

implementation is heavily inspired by the slim.datasets implementation (https://github.com/tensorflow/models/tree/master/research/slim/datasets)
"""
import os
import sys

import numpy as np
from six.moves import urllib
import gzip
import zipfile
import tensorflow as tf

import src.data.util_data as util_data

# The URLs where the MNIST data can be downloaded.
_DATA_URL = 'https://vision.eng.au.dk/?download=/data/WeedData/'
_NONSEGMENTED = 'Nonsegmented.zip'
_SEGMENTED = 'Segmented.zip'

# Local directories to store the dataset
_DIR_RAW = 'data/raw/PSD'
_DIR_PROCESSED = 'data/processed/PSD'

#
_IMAGE_SIZE = 28
_NUM_CHANNELS = 1
_NUM_TRAIN_SAMPLES = 60000
_NUM_TEST_SAMPLES = 10000



class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB PNG data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)
        self._encode_png = tf.image.encode_png(self._decode_png)

    def truncate_image(self, sess, image_data):
        image, reencoded_image = sess.run(
            [self._decode_png, self._encode_png],
            feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return reencoded_image, image.shape[0], image.shape[1], image.shape[2]

    def read_image_dims(self, sess, image_data):
        image = self.decode_png(sess, image_data)
        return image.shape[0], image.shape[1], image.shape[2]

    def decode_png(self, sess, image_data):
        image = sess.run(self._decode_png,
            feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def encode_png(self, sess, image_data):
        image_data = sess.run(self._encode_png,
            feed_dict={self._decode_png_data: image_data})


def _get_filenames_and_classes(dataset_dir, setname):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    flower_root = os.path.join(dataset_dir, setname)
    directories = []
    class_names = []
    for filename in os.listdir(flower_root):
        path = os.path.join(flower_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _convert_to_tfrecord(filenames, class_dict, tfrecord_writer):
    """Loads data from the binary MNIST files and writes files to a TFRecord.

    Args:
        data_filename: The filename of the MNIST images.
        labels_filename: The filename of the MNIST labels.
        num_images: The number of images in the dataset.
        tfrecord_writer: The TFRecord writer to use for writing.
    """
    
    num_images = len(filenames)

    image_reader = ImageReader()

    with tf.Session('') as sess:
        for i in range(num_images):
            sys.stdout.write('\r>> Converting image %d/%d' % (i + 1, num_images))
            sys.stdout.flush()

            # Read the filename:
            encoded_img = tf.gfile.FastGFile(filenames[i], 'rb').read()
            encoded_img, height, width, channels = image_reader.truncate_image(sess, encoded_img)#  .read_image_dims(sess, encoded_img)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            label = class_dict[class_name]

            example = util_data.encode_image(
                image_data = encoded_img,
                image_format = 'png'.encode(),
                class_lbl = label,
                class_text = class_name.encode(),
                height = height,
                width = width,
                channels = channels,
                origin = filenames[i].encode()
                )

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
  """Downloads PSD locally.
  """
  for filename in [_NONSEGMENTED,
                   _SEGMENTED]:
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

    training_filename = _get_output_filename(_DIR_PROCESSED, 'Nonsegmented')
    testing_filename = _get_output_filename(_DIR_PROCESSED, 'Segmented')

    if tf.gfile.Exists(training_filename) and tf.gfile.Exists(testing_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return


    # First, process the nonsegented data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        data_filename = os.path.join(_DIR_RAW, _NONSEGMENTED)
        archive = zipfile.ZipFile(data_filename)
        archive.extractall(_DIR_PROCESSED)
        filenames, class_names = _get_filenames_and_classes(_DIR_PROCESSED, 'Nonsegmented')
        class_dict = dict(zip(class_names, range(len(class_names))))

        _convert_to_tfrecord(filenames, class_dict, tfrecord_writer)

        tmp_dir = os.path.join(_DIR_PROCESSED, 'Nonsegmented')
        tf.gfile.DeleteRecursively(tmp_dir)

    # Next, process the segmented data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        data_filename = os.path.join(_DIR_RAW, _SEGMENTED)
        archive = zipfile.ZipFile(data_filename)
        archive.extractall(_DIR_PROCESSED)
        filenames, class_names = _get_filenames_and_classes(_DIR_PROCESSED, 'Segmented')
        class_dict = dict(zip(class_names, range(len(class_names))))

        _convert_to_tfrecord(filenames, class_dict, tfrecord_writer)

        tmp_dir = os.path.join(_DIR_PROCESSED, 'Segmented')
        tf.gfile.DeleteRecursively(tmp_dir)


    print('\nFinished converting the PSD dataset!')
