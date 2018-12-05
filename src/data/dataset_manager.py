import sys
import os

import src.utils as utils
import src.data.util_data as util_data

import src.data.datasets.mnist as mnist
import src.data.datasets.psd as psd

def make_dataset(dataset):
    dir_rawData = 'data/raw/'+ dataset
    utils.checkfolder(dir_rawData)

    if dataset == 'MNIST':
        mnist.download()

    elif dataset == 'PSD_Nonsegmented': 
        psd.download('Nonsegmented')

    elif dataset == 'PSD_Segmented':
        psd.download('Segmented')

    else:
        pass

def process_dataset(dataset):
    dir_processedData = 'data/processed/'+ dataset
    utils.checkfolder(dir_processedData)

    if dataset == 'MNIST':
        mnist.process()

    elif dataset == 'PSD_Nonsegmented':
        psd.process('Nonsegmented')

    elif dataset == 'PSD_Segmented':
        psd.process('Segmented')

    else:
        pass
