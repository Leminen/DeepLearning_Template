#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:43:52 2017

@author: leminen
"""
import os
import tensorflow as tf
import src.models.ops_util as ops
import src.utils as utils


class BasicModel(object):
    def __init__(self):
        self.model = 'BasicModel'
        self.dir_logs        = 'models/' + self.model + '/logs'
        self.dir_checkpoints = 'models/' + self.model + '/checkpoints'
        self.dir_results     = 'models/' + self.model + '/results'
        
        utils.checkfolder(self.dir_checkpoints)
        utils.checkfolder(self.dir_logs)
        utils.checkfolder(self.dir_results)
       
    def _create_inference(self):
        """ Define the inference model for the network
        Args:
    
        Returns:
        """
        ### self.output = f(self.input) ## define f
    
    def _create_losses(self):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """
        ### self.loss = f(self.output, self.input) ## define f
        
    def _create_optimizer(self):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """
        ### self.optimizer_op = f(self.loss) ## define f
        
    def _create_summaries(self):
        """ Create summaries for the network
        Args:
    
        Returns:
        """
        
        ### Add summaries
        
        self.summary_op = tf.summary.merge_all()
        
        
    def train(self, dataset_str, epoch_N, batch_N):
        """ Run training of the network
        Args:
    
        Returns:
        """
        
        # Use dataset for loading in datasamples from .tfrecord (https://www.tensorflow.org/programmers_guide/datasets#consuming_tfrecord_data)
        # The iterator will get a new batch from the dataset each time a sess.run() is executed on the graph.
        filenames = tf.placeholder(tf.string, shape=[None])
        
        dataset = tf.contrib.data.TFRecordDataset(filenames)
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.batch(batch_size = batch_N)
        iterator = dataset.make_initializable_iterator()
        self.input = iterator.get_next()
        
        # define model, loss, optimizer and summaries.
        self._create_inference()
        self._create_losses()
        self._create_optimizer()
        self._create_summaries()
        
        
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
            
            counter = 0
            
            # Do training loops
            for epoch_n in range(epoch_start, epoch_N):
                
                training_filenames = ["/data/dataset/file1.tfrecord", "/data/dataset/file2.tfrecord"] # EXAMPLE !
                sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
                
                while True:
                    try:
                        _, summary = sess.run([self.optimizer_op, self.summary_op])
                        
                        writer.add_summary(summary, global_step=counter)
                        counter =+ 1
                        
                    except tf.errors.OutOfRangeError:
                        break
                
                
                if epoch_n % 1 == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)

                
            
            
    
    def predict(self):
        """ Run prediction of the network
        Args:
    
        Returns:
        """
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    
