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

import src.utils as utils
import src.data.util_data as util_data

# The URLs where the PSD data can be downloaded.
_DATA_URL = 'https://vision.eng.au.dk/?download=/data/WeedData/'
_NONSEGMENTED = 'NonsegmentedV2.zip'
_SEGMENTED = 'Segmented.zip'

_DATA_URL_NONSEGMENTED = 'https://vision.eng.au.dk/?download=/data/WeedData/NonsegmentedV2.zip'
_DATA_URL_SEGMENTED = 'https://vision.eng.au.dk/?download=/data/WeedData/Segmented.zip'

# Local directories to store the dataset
_DIR_RAW = 'data/raw/PSD'
_DIR_PROCESSED = 'data/processed/PSD'

_DIR_RAW_NONSEGMENTED = 'data/raw/PSD_Nonsegmented/NonsegmentedV2.zip'
_DIR_PROCESSED_NONSEGMENTED = 'data/processed/PSD_Nonsegmented/'

_DIR_RAW_SEGMENTED = 'data/raw/PSD_Segmented/Segmented.zip'
_DIR_PROCESSED_SEGMENTED = 'data/processed/PSD_Segmented/'


_EXCLUDED_GRASSES = True
_EXCLUDE_LARGE_IMAGES = True
_LARGE_IMAGE_DIM = 400
_NUM_SHARDS = 10


def chunkify(lst,n):
    return [lst[i::n] for i in iter(range(n))]

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


def _get_filenames_and_classes(dataset_dir, setname, exclude_list):
    """Returns a list of filenames and inferred class names.

    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    
    data_root = os.path.join(dataset_dir, *setname)

    directories = []
    class_names = []
    for filename in os.listdir(data_root):
        path = os.path.join(data_root, filename)
        if os.path.isdir(path):
            if not any(x in filename for x in exclude_list):
                directories.append(path)
                class_names.append(filename)

    photo_filenames = []
    photo_filenames2 = []
    for _ in range(_NUM_SHARDS):
        photo_filenames2.append([])

    for directory in directories:
        if not any(x in directory for x in exclude_list):
            filenames = os.listdir(directory)
            paths = [os.path.join(directory, filename) for filename in filenames]
            paths_split = chunkify(paths,_NUM_SHARDS)

            for shard_n in range(_NUM_SHARDS):
                photo_filenames2[shard_n].extend(paths_split[shard_n])

            for filename in filenames:
                path = os.path.join(directory, filename)
                photo_filenames.append(path)
                

    return photo_filenames2, sorted(class_names)


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
            encoded_img, height, width, channels = image_reader.truncate_image(sess, encoded_img)

            if _EXCLUDE_LARGE_IMAGES and (height > _LARGE_IMAGE_DIM or width > _LARGE_IMAGE_DIM):
                pass
            else:
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
        

def _get_output_filename(dataset_dir, shard_id):
    """Creates the output filename.

    Args:
      dataset_dir: The directory where the temporary files are stored.
      split_name: The name of the train/test split.

    Returns:
      An absolute file path.
    """
    return '%s/PSD-data_%03d-of-%03d.tfrecord' % (dataset_dir, shard_id+1, _NUM_SHARDS)


def download(dataset_part):
    """Downloads PSD locally
    """
    if dataset_part == 'Nonsegmented':
        _data_url = _DATA_URL_NONSEGMENTED
        filepath = os.path.join(_DIR_RAW_NONSEGMENTED)
    else:
        _data_url = _DATA_URL_SEGMENTED
        filepath = os.path.join(_DIR_RAW_SEGMENTED)

    if not os.path.exists(filepath):
        print('Downloading dataset...')
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %.1f%%' % (
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(_data_url, filepath, _progress)

        print()
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', size, 'bytes.')



def process(dataset_part):
    """Runs the download and conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if dataset_part == 'Nonsegmented':
        _dir_raw = _DIR_RAW_NONSEGMENTED
        _dir_processed = _DIR_PROCESSED_NONSEGMENTED
        setname = 'Nonsegmented'
        #training_filename = _get_output_filename(_DIR_PROCESSED_NONSEGMENTED, 'train')
        # testing_filename = _get_output_filename(_DIR_PROCESSED_NONSEGMENTED, 'test')
    else:
        _dir_raw = _DIR_RAW_SEGMENTED
        _dir_processed = _DIR_PROCESSED_SEGMENTED
        setname = 'Segmented' 
        #training_filename = _get_output_filename(_DIR_PROCESSED_SEGMENTED, 'train')
        # testing_filename = _get_output_filename(_DIR_PROCESSED_SEGMENTED, 'test')

    #if tf.gfile.Exists(training_filename): #and tf.gfile.Exists(testing_filename):
    #    print('Dataset files already exist. Exiting without re-creating them.')
    #    return


    if _EXCLUDED_GRASSES:
        exclude_list = ['Black-grass', 'Common wheat', 'Loose Silky-bent']
    else:
        exclude_list = []

    # First, process training data:

    data_filename = os.path.join(_dir_raw)
    archive = zipfile.ZipFile(data_filename)
    archive.extractall(_dir_processed)
    filenames, class_names = _get_filenames_and_classes(_dir_processed, [setname], exclude_list)

    class_dict = dict(zip(class_names, range(len(class_names))))
    utils.save_dict(class_dict, _dir_processed, 'class_dict.json')

    for shard_n in range(_NUM_SHARDS):
        utils.show_message('Processing shard %d/%d' % (shard_n+1,_NUM_SHARDS))
        tf_filename = _get_output_filename(_dir_processed, shard_n)

        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            _convert_to_tfrecord(filenames[shard_n], class_dict, tfrecord_writer)

    tmp_dir = os.path.join(_dir_processed, setname)
    tf.gfile.DeleteRecursively(tmp_dir)

    # # First, process test data:
    # with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
    #     data_filename = os.path.join(_dir_raw)
    #     archive = zipfile.ZipFile(data_filename)
    #     archive.extractall(_dir_processed)
    #     # filenames, class_names = _get_filenames_and_classes(_dir_processed, [setname, 'test'], exclude_list)
    #     class_dict = dict(zip(class_names, range(len(class_names))))

    #     _convert_to_tfrecord(filenames, class_dict, tfrecord_writer)

    #     tmp_dir = os.path.join(_dir_processed, setname)
    #     tf.gfile.DeleteRecursively(tmp_dir)

    print('\nFinished converting the PSD %s dataset!' % setname)



  


    
