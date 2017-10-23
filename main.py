"""
This file is used to run the project.
Notes:
- The structure of this file (and the entire project in general) is made with emphasis on flexibility for research
purposes, and the pipelining is done in a python file such that newcomers can easily use and understand the code.
- Remember that relative paths in Python are always relative to the current working directory.

Hence, if you look at the functions in make_dataset.py, the file paths are relative to the path of
this file (main.py)
"""

__author__ = "Simon Leminen Madsen"
__email__ = "slm@eng.au.dk"

import os
import argparse

from src.data import make_dataset
from src.data import process_dataset
from src.models.BasicModel import BasicModel
from src.visualization import visualize


"""parsing and configuration"""
def parse_args():
    
# ----------------------------------------------------------------------------------------------------------------------
# Define default pipeline
# ----------------------------------------------------------------------------------------------------------------------

    desc = "Pipeline for running Tensorflow implementation of infoGAN"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--make_dataset', 
                        action='store_true', 
                        help = 'Fetch dataset from remote source into /data/raw/. Or generate raw dataset [Defaults to False if argument is omitted]')
    
    parser.add_argument('--process_dataset', 
                        action='store_true', 
                        help = 'Run preprocessing of raw data. [Defaults to False if argument is omitted]')

    parser.add_argument('--train_model', 
                        action='store_true', 
                        help = 'Run configuration and training network [Defaults to False if argument is omitted]')

    parser.add_argument('--visualize', 
                        action='store_true', 
                        help = 'Run visualization of results [Defaults to False if argument is omitted]')
    
# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments used in the entire pipeline
# ----------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--model', 
                        type=str, 
                        default='BasicModel', 
                        choices=['BasicModel'],
                        required = True,
                        help='The name of the network model')

    parser.add_argument('--dataset', 
                        type=str, default='mnist', 
                        choices=['mnist'],
                        required = True,
                        help='The name of dataset')
    
# ----------------------------------------------------------------------------------------------------------------------
# Define the arguments for the training
# ----------------------------------------------------------------------------------------------------------------------

    parser.add_argument('--hparams',
                        type=str,
                        help='Comma separated list of "name=value" pairs.')


    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    
    # Assert if training parameters are provided, when training is selected
#    if args.train_model:
#        try:
#            assert args.hparams is ~None
#        except:
#            print('hparams not provided for training')
#            exit()
        
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    
    # Make dataset
    if args.make_dataset:
        print('Fetching raw dataset: ' + args.dataset)
        make_dataset.make_dataset(args.dataset)
        
    # Make dataset
    if args.process_dataset:
        print('Processing raw dataset: ' + args.dataset)
        process_dataset.process_dataset(args.dataset)
        #################################
        ####### To Be Implemented #######
        #################################
        
    # Build and train model
    if args.train_model:
        print('Configuring and training Network: '+ args.model)
        
        if args.model == 'BasicModel':
            model = BasicModel()
            model.train(dataset_str = args.dataset, epoch_N = 1, batch_N = 10)
    
    # Visualize results
    if args.visualize:
        print('Visualizing Results')
        #################################
        ####### To Be Implemented #######
        #################################
    

if __name__ == '__main__':
    main()
