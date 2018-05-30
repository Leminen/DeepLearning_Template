"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""

import os
import datetime
import tensorflow as tf
import tensorflow.contrib.slim as slim


def checkfolder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def show_message(msg_str, lvl=0):

    if lvl == 0:
        print(datetime.datetime.now(), '-', msg_str)
    elif lvl == 1:
        print('______________________________________________________________')
        print(datetime.datetime.now(), '-', msg_str)
        print('--------------------------------------------------------------')
    else:
        pass
        
    

