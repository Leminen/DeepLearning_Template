# -*- coding: utf-8 -*-
import os
from tensorflow.contrib.learn.python.learn.datasets import mnist

import src.utils as utils

def make_dataset(dataset):
    dir_rawData = 'data/raw/'+ dataset
    utils.checkfolder(dir_rawData)
    
    if dataset == 'mnist':
        # Download the MNIST dataset from source and save it in 'data/raw/mnist'
        data = mnist.read_data_sets('data/raw/mnist', one_hot=True) 

