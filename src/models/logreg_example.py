#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:43:52 2017

@author: leminen
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import argparse
import shlex

import src.utils as utils
import src.data.util_data as util_data


def hparams_parser_train(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_max', 
                        type=int, default='100', 
                        help='Max number of epochs to run')

    parser.add_argument('--batch_size', 
                        type=int, default='64', 
                        help='Number of samples in each batch')

    ## add more model parameters to enable configuration from terminal
    
    return parser.parse_args(shlex.split(hparams_string))


def hparams_parser_evaluate(hparams_string):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch_no', 
                        type=int,
                        default=None, 
                        help='Epoch no to reload')

    ## add more model parameters to enable configuration from terminal

    return parser.parse_args(shlex.split(hparams_string))


class logreg_example(object):
    def __init__(self, dataset, id):

        self.model = 'logreg_example'
        if id != None:
            self.model = self.model + '_' + id

        self.dir_base        = 'models/' + self.model
        self.dir_logs        = self.dir_base + '/logs'
        self.dir_checkpoints = self.dir_base + '/checkpoints'
        self.dir_results     = self.dir_base + '/results'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)

        # Specify valid dataset for model
        if dataset == 'MNIST':
            self.dateset_filenames =  ['data/processed/MNIST/train.tfrecord']
            self.lbl_dim = 10
        else:
            raise ValueError('Selected Dataset is not supported by model: logreg_example')
        
       
    def _create_inference(self, inputs):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        X = tf.reshape(inputs,[-1,784])
        
        w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
        b = tf.Variable(tf.zeros([1, 10]), name="bias")
        
        outputs = tf.matmul(X, w) + b 
        return outputs
    
    def _create_losses(self, outputs, labels):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """

        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels, name='loss')
        loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch
        return loss
        
    def _create_optimizer(self, loss):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """

        optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
        optimizer_op = optimizer.minimize(loss)
        return optimizer_op
        
    def _create_summaries(self, loss):
        """ Create summaries for the network
        Args:
    
        Returns:
        """
        
        ### Add summaries
        with tf.name_scope("summaries"):
            tf.summary.scalar('model_loss', loss) # placeholder summary
            summary_op = tf.summary.merge_all()

        return summary_op
        
        
    def train(self, hparams_string):
        """ Run training of the network
        Args:
    
        Returns:
        """
        args_train = hparams_parser_train(hparams_string)
        batch_size = args_train.batch_size
        epoch_max = args_train.epoch_max

        utils.save_model_configuration(args_train, self.dir_base)
        
        # Use dataset for loading in datasamples from .tfrecord (https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)
        # The iterator will get a new batch from the dataset each time a sess.run() is executed on the graph.
        dataset = tf.data.TFRecordDataset(self.dateset_filenames)
        dataset = dataset.map(util_data.decode_image)      # decoding the tfrecord
        dataset = dataset.map(self._preProcessData)        # potential local preprocessing of data
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.batch(batch_size = batch_size)
        iterator = dataset.make_initializable_iterator()
        inputs = iterator.get_next()

        # depends on self._preProcessData
        [in_image, in_label] = inputs

        # show network architecture
        utils.show_all_variables()
        
        # define model, loss, optimizer and summaries.
        outputs = self._create_inference(in_image)
        loss = self._create_losses(outputs, in_label)
        optimizer_op = self._create_optimizer(loss)
        summary_op = self._create_summaries(loss)
        
        
        with tf.Session() as sess:
            
            # Initialize all model Variables.
            sess.run(tf.global_variables_initializer())
            
            # Create Saver object for loading and storing checkpoints
            saver = tf.train.Saver()
            
            # Create Writer object for storing graph and summaries for TensorBoard
            writer = tf.summary.FileWriter(self.dir_logs, sess.graph)
            
            
            # Reload Tensor values from latest checkpoint
            ckpt = tf.train.get_checkpoint_state(self.dir_checkpoints)
            epoch_start = 0
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                epoch_start = int(ckpt_name.split('-')[-1])
            
            interationCnt = 0
            # Do training loops
            for epoch_n in range(epoch_start, epoch_max):

                # Initiate or Re-initiate iterator
                sess.run(iterator.initializer)

                # Test model output before any training
                if epoch_n == 0:
                    summary = sess.run(summary_op)
                    writer.add_summary(summary, global_step=-1)

                utils.show_message('Running training epoch no: {0}'.format(epoch_n))
                while True:
                    try:
                        _, summary = sess.run([optimizer_op, summary_op])
                        
                        writer.add_summary(summary, global_step=interationCnt)
                        counter =+ 1
                        
                    except tf.errors.OutOfRangeError:
                        # Do some evaluation after each Epoch
                        break
                
                if epoch_n % 1 == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)
                
            
    
    def evaluate(self, hparams_string):
        """ Run prediction of the network
        Args:
    
        Returns:
        """
        
        args_evaluate = hparams_parser_evaluate(hparams_string)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    

    def _preProcessData(self, image_proto, lbl_proto, class_proto, height_proto, width_proto, channels_proto, origin_proto):
        """ Local preprocessing of data from dataset
        also used to select which elements to parse onto the model
        Args:
          all outputs of util_data.decode_image

        Returns:
        """
        image = image_proto
        label = tf.one_hot(lbl_proto, self.lbl_dim)

        return image, label