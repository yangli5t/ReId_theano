__author__ = 'Saizheng Zhang'

import numpy as np
import scipy.io
import theano
from theano import tensor as T
from Layer_new import Conv, Fully, Pooling, Activation, Flatten, Concat, Model, Transformer, Dropout, Abs_diff
from Layer_new import CostFunc
from keras.optimizers import SGD, RMSprop
from keras.constraints import identity
from keras.regularizers import l1
import pdb

class V0(object):
    def __init__(self):
        X1 = T.tensor4()
        X2 = T.tensor4()
        X = [X1, X2]
        Y = [T.ivector()]
        
        model = Model()
        #conv1
        model.add(Conv(filter_shape = (25, 3, 5, 5), w_shared = True, n_inputs = 2))
        model.add(Pooling(pool_size = (2,2)))
        model.add(Activation(mode = 'tanh'))
        #conv2
        model.add(Conv(filter_shape = (25, 25, 3, 3), w_shared = True, n_inputs = 2))
        model.add(Pooling(pool_size = (2,2)))
        model.add(Activation(mode = 'tanh'))
        #abs_diff
        model.add(Abs_diff())
        #conv3
        model.add(Conv(filter_shape = (25, 25, 3, 3), w_shared = True))
        model.add(Pooling(pool_size = (2,2)))
        model.add(Activation(mode = 'tanh'))
        model.add(Flatten())
        model.add(Fully((25*18*5, 500)))
        model.add(Activation(mode = 'tanh'))
        model.add(Fully((500, 2)))
        model.add(Activation(mode = 'softmax'))
        model.build(CostFunc.nll, RMSprop(), X, Y)
        self.model = model

class V1(object):
    def __init__(self):
        X1 = T.tensor4()
        X2 = T.tensor4()
        X = [X1, X2]
        Y = [T.ivector()]
        
        model = Model()
        #conv1
        model.add(Conv(filter_shape = (32, 3, 3, 3), w_shared = True, n_inputs = 2))
        model.add(Conv(filter_shape = (32, 32, 2, 2), w_shared = True, n_inputs = 2))
        model.add(Pooling(pool_size = (2,2)))
        model.add(Activation(mode = 'tanh'))
        #conv2
        model.add(Conv(filter_shape = (32, 32, 3, 3), w_shared = True, n_inputs = 2))
        model.add(Pooling(pool_size = (2,2)))
        model.add(Activation(mode = 'tanh'))
        #abs_diff
        model.add(Abs_diff())
        #conv3
        model.add(Conv(filter_shape = (32, 32, 3, 3), w_shared = True))
        model.add(Pooling(pool_size = (2,2)))
        model.add(Activation(mode = 'tanh'))
        model.add(Flatten())

        self.f = theano.function(X, model.f(X, is_train = True))

        model.add(Fully((2880, 512)))
        model.add(Activation(mode = 'tanh'))
        model.add(Fully((512, 2)))
        model.add(Activation(mode = 'softmax'))
        model.build(CostFunc.nll, RMSprop(), X, Y)
        self.model = model

class V1_reg(object):
    def __init__(self):
        X1 = T.tensor4()
        X2 = T.tensor4()
        X = [X1, X2]
        Y = [T.ivector()]
        
        model = Model()
        #conv1
        model.add(Conv(filter_shape = (32, 3, 3, 3), regularizers = {'W': l1(0.0001)},  w_shared = True, n_inputs = 2))
        model.add(Conv(filter_shape = (32, 32, 2, 2), regularizers = {'W': l1(0.0001)}, w_shared = True, n_inputs = 2))
        model.add(Pooling(pool_size = (2,2)))
        model.add(Activation(mode = 'tanh'))
        #conv2
        model.add(Conv(filter_shape = (32, 32, 3, 3), regularizers = {'W': l1(0.0001)}, w_shared = True, n_inputs = 2))
        model.add(Pooling(pool_size = (2,2)))
        model.add(Activation(mode = 'tanh'))
        #abs_diff
        model.add(Abs_diff())
        #conv3
        model.add(Conv(filter_shape = (32, 32, 3, 3), regularizers = {'W': l1(0.0001)}, w_shared = True))
        model.add(Pooling(pool_size = (2,2)))
        model.add(Activation(mode = 'tanh'))
        model.add(Flatten())

        self.f = theano.function(X, model.f(X, is_train = True))

        model.add(Fully((2880, 512)))
        model.add(Activation(mode = 'tanh'))
        model.add(Dropout(0.5))
        model.add(Fully((512, 2)))
        model.add(Activation(mode = 'softmax'))
        model.build(CostFunc.nll, RMSprop(), X, Y)
        self.model = model


if __name__ == "__main__":
    x1 = np.ones((10, 3, 160, 60))
    x2 = 2*np.ones((10, 3, 160, 60))
    model = V1()
    f = model.f
    out = f(x1, x2)
    pdb.set_trace()
