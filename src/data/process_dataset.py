
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import mnist

import src.utils as utils

def process_dataset(dataset):
    dir_processedData = 'data/processed/'+ dataset
    utils.checkfolder(dir_processedData)
    
    if dataset == 'mnist':
        # Download the MNIST dataset from source and save it in 'data/raw/mnist'
        data = mnist.read_data_sets('data/raw/mnist', reshape=False)
        _convert_to_tfrecord(data.train, dataset, 'train')
        _convert_to_tfrecord(data.validation, dataset, 'validation')
        _convert_to_tfrecord(data.test, dataset, 'test')
        


def __int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def __bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_tfrecord(data, dataset, name):
    """Converts a dataset to tfrecords."""
    images = data.images
    labels = data.labels
    num_examples = data.num_examples

    if images.shape[0] != num_examples:
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], num_examples))

    filename = os.path.join('data/processed/',dataset, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
      
    for index in range(num_examples):
        example = _encodeData(images[index], labels[index])

        writer.write(example.SerializeToString())
    writer.close()
    
### Define data encoder and decoder for the .tfrecord file[s]. The metodes must be reverse of each other,
### Encoder will be used by process_dataset directly whereas the decoder is used by the Model[s] to load data
### Look at this guide to format the tfrecord features: http://www.machinelearninguru.com/deep_learning/tensorflow/basics/tfrecord/tfrecord.html
    
def _encodeData(image,lbl):
    image_raw = image.tostring()
    shape = np.array(image.shape, np.int32)
    shape = shape.tobytes()
    
    features = {'label'     : __int64_feature(int(lbl)),
                'shape'     : __bytes_feature(shape),
                'image_raw' : __bytes_feature(image_raw)}
    
    example = tf.train.Example(features=tf.train.Features(feature=features))
    
    return example

def _decodeData(example_proto):
    features = {'label'     : tf.FixedLenFeature([], tf.int64),
                'shape'     : tf.FixedLenFeature([], tf.string),
                'image_raw' : tf.FixedLenFeature([], tf.string)}
   
    parsed_features = tf.parse_single_example(example_proto, features)

    shape = tf.decode_raw(parsed_features['shape'], tf.int32)
    image = tf.decode_raw(parsed_features['image_raw'], tf.float32)
    image = tf.reshape(image, shape)

    label = parsed_features['label']
    
    return image, label
    
    
    