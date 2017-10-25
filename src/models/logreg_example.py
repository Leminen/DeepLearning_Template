#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:43:52 2017

@author: leminen
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt

import src.data.process_dataset as process_dataset
import src.models.ops_util as ops
import src.utils as utils


class Logreg_example(object):
    def __init__(self):
        self.model = 'Logreg_example'
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
        
        X = tf.reshape(self.input_images,[-1,784])
        
        w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
        b = tf.Variable(tf.zeros([1, 10]), name="bias")
        
        self.output = tf.matmul(X, w) + b 
        
    
    def _create_losses(self):
        """ Define loss function[s] for the network
        Args:
    
        Returns:
        """
        ### self.loss = f(self.output, self.input) ## define f
        
        Y = tf.one_hot(self.input_labels, depth = 10)
        
        entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=Y, name='loss')
        self.loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch
        
    def _create_optimizer(self):
        """ Create optimizer for the network
        Args:
    
        Returns:
        """
        ### self.optimizer_op = f(self.loss) ## define f
        self.optimizer_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(self.loss)

    def _create_summaries(self):
        """ Create summaries for the network
        Args:
    
        Returns:
        """
        
        ### Add summaries
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss) 
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
        dataset = dataset.map(process_dataset._decodeData)      # decoding the tfrecord
        dataset = dataset.shuffle(buffer_size = 10000, seed = None)
        dataset = dataset.batch(batch_size = batch_N)
        iterator = dataset.make_initializable_iterator()
        self.input_images, self.input_labels = iterator.get_next()
        
        print('Dataset output shape: ', dataset.output_shapes, 'Dataset output types: ',dataset.output_types)
        
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
                
                training_filenames = ['data/processed/' + dataset_str + '/train.tfrecords'] # EXAMPLE
                sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
                
                print('Running training epoch no: ', epoch_n)
                while True:
                    try:
                        _, summary = sess.run([self.optimizer_op, self.summary_op])
                        
                        writer.add_summary(summary, global_step=counter)
                        counter =+ 1
                        
                    except tf.errors.OutOfRangeError:
                        break
                
                if epoch_n % 1 == 0:
                    saver.save(sess,os.path.join(self.dir_checkpoints, self.model + '.model'), global_step=epoch_n)
                
                
                ### TEST of Input
#                for _ in range(10):
#                    input_imgs, input_lbls = sess.run(self.input_images, self.input_labels)
#                            
#                    print('Label = ', input_lbls, 'Input Data Shape = ', input_imgs.shape, 'Plotting first image!')
#                    plt.imshow(input_imgs[0].squeeze())
#                    plt.show()
            
    
    def predict(self):
        """ Run prediction of the network
        Args:
    
        Returns:
        """
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
    

        
        