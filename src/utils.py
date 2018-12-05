import os
import datetime
import json
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy


class Namespace(dict):
    """A dict subclass that exposes its items as attributes.

    Warning: Namespace instances do not have direct access to the
    dict methods.

    Taken from: http://code.activestate.com/recipes/577887-a-simple-namespace-class/

    """

    def __init__(self, obj={}):
        super().__init__(obj)

    def __dir__(self):
        return tuple(self)

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, super().__repr__())

    def __getattribute__(self, name):
        try:
            return self[name]
        except KeyError:
            msg = "'%s' object has no attribute '%s'"
            raise AttributeError(msg % (type(self).__name__, name))

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    #------------------------
    # "copy constructors"

    @classmethod
    def from_object(cls, obj, names=None):
        if names is None:
            names = dir(obj)
        ns = {name:getattr(obj, name) for name in names}
        return cls(ns)

    @classmethod
    def from_mapping(cls, ns, names=None):
        if names:
            ns = {name:ns[name] for name in names}
        return cls(ns)

    @classmethod
    def from_sequence(cls, seq, names=None):
        if names:
            seq = {name:val for name, val in seq if name in names}
        return cls(seq)

    #------------------------
    # static methods

    @staticmethod
    def hasattr(ns, name):
        try:
            object.__getattribute__(ns, name)
        except AttributeError:
            return False
        return True

    @staticmethod
    def getattr(ns, name):
        return object.__getattribute__(ns, name)

    @staticmethod
    def setattr(ns, name, value):
        return object.__setattr__(ns, name, value)

    @staticmethod
    def delattr(ns, name):
        return object.__delattr__(ns, name)




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

def save_dict(dict, path, filename):
    fullpath = os.path.join(path, filename)
    with open(fullpath, 'w+') as fp:
        json.dump(dict, fp)

def save_model_configuration(args, path):
    args_dict = vars(args)
    filename = 'configuration.json'
    save_dict(args_dict,path,filename)

def load_model_configuration(path):
    fullpath = os.path.join(path, 'configuration.json')
    with open(fullpath, 'r') as fp:
        data = json.load(fp)
    
    return Namespace(data)

def save_image_local(image, path, infostr):
    datestr = datetime.datetime.now().strftime('%Y%m%d_T%H%M%S')
    filename = datestr + '_' + infostr + '.png'
    path = os.path.join(path,filename)
    image = np.squeeze(image)
    scipy.misc.toimage(image, cmin=-1, cmax=1).save(path)

def save_image_local_batch(images,path,infostr):
    n_images = images.shape[0]
    for i in range(n_images):
        save_image_local(images[i,:,:,:],path, infostr + '_' +str(i))
