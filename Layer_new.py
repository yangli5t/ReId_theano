__author__ = 'Saizheng Zhang'
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.tensor.nnet import neighbours
from theano.tensor.shared_randomstreams import RandomStreams
from keras import initializations, activations
from keras.utils.theano_utils import shared_zeros
from keras.constraints import identity
from copy import deepcopy
import pdb
from collections import OrderedDict

class Layer(object):
    def __init__(self):
        self.name = self.__class__.__name__

    # currently for the same parameter in the same layer,
    # the regularizers and constraints are the same among different inputs.
    def set_regularizers(self, regularizers = None):
        if regularizers:
            self.regularizers = []
            for name in regularizers:
                for param in self.params_dict[name]:
                    r = deepcopy(regularizers[name])
                    r.set_param(param)
                    self.regularizers.append(r)

    def set_constraints(self, constraints = None):
        self.constraints = []
        for name in self.params_dict.keys():
            if not constraints:
               self.constraints += [identity() for param in self.params_dict[name]]
            elif constraints.has_key(name):
               self.constraints += [deepcopy(constraints[name]) \
                                    for param in self.params_dict[name]]
            else:
               self.constraints += [identity() for param in self.params_dict[name]]


class Conv(Layer):
    def __init__(self, filter_shape, b_mode = 'valid', init_mode = 'glorot_uniform', w_shared = True, n_inputs = 1, regularizers = None, constraints = None):
        self.name = self.__class__.__name__
        self.init = initializations.get(init_mode)
        self.w_shared = w_shared
        self.b_mode = b_mode 
        self.filter_shape = filter_shape

        self.params_dict = OrderedDict( \
            [('W', [self.init(filter_shape)] if w_shared \
                   else [self.init(filter_shape) for i in xrange(n_inputs)]),
             ('b', [shared_zeros((filter_shape[0],))] if w_shared \
                   else [shared_zeros((filter_shape[0],)) for i in xrange(n_inputs)])])
        self.params = [param for sublist in self.params_dict.values() for param in sublist]
        self.set_constraints(constraints)
        self.set_regularizers(regularizers)
       
    def f(self, inputs):
        return [conv.conv2d(input = inputs[i],
                            filters = self.params_dict['W'][0] \
                                if self.w_shared else self.params_dict['W'][i],
                            filter_shape = self.filter_shape, border_mode = self.b_mode) + \
                (self.params_dict['b'][0].dimshuffle('x',0,'x','x') \
                 if self.w_shared else self.params_dict['b'][i].dimshuffle('x',0,'x','x')) \
                for i in xrange(len(inputs))]


class Fully(Layer):
    def __init__(self, filter_shape, init_mode = 'glorot_uniform', w_shared = True, n_inputs = 1, regularizers = None, constraints = None):
        self.name = self.__class__.__name__
        self.init = initializations.get(init_mode)
        self.w_shared = w_shared
        self.filter_shape = filter_shape

        self.params_dict = OrderedDict( \
            [('W', [self.init(filter_shape)] if w_shared \
                   else [self.init(filter_shape) for i in xrange(n_inputs)]),
             ('b', [shared_zeros((filter_shape[1],))] if w_shared \
                   else [shared_zeros((filter_shape[1],)) for i in xrange(n_inputs)])])
        self.params = [param for sublist in self.params_dict.values() for param in sublist]
        self.set_constraints(constraints)
        self.set_regularizers(regularizers)

    def f(self, inputs):
        return [T.dot(inputs[i], (self.params_dict['W'][0] if self.w_shared else self.params_dict['W'][i])) + \
                (self.params_dict['b'][0] if self.w_shared else self.params_dict['b'][i]) \
                for i in xrange(len(inputs))]


class Pooling(Layer):
    def __init__(self, pool_size = (2,2), mode = 'max_pooling'):
        self.name = self.__class__.__name__ 
        if mode is not 'max_pooling':
            raise NotImplementedError
        self.pool_size = pool_size

    def f(self, inputs):
        return  [downsample.max_pool_2d(input=x, ds=self.pool_size, ignore_border=True) \
                 for x in inputs]


class Activation(Layer):
    # currently Activation does not support parameterization.
    def __init__(self, mode = 'relu'):
        self.name = self.__class__.__name__ + mode
        self.nonlinearty = activations.get(mode)

    def f(self, inputs):
        return [self.nonlinearty(x) for x in inputs]


class Flatten(Layer):
    def __init__(self):
        self.name = self.__class__.__name__ 

    def f(self, inputs):
        return [x.flatten(ndim=2) for x in inputs]


class Concat(Layer):
    def __init__(self):
        self.name = self.__class__.__name__ 

    def f(self, inputs):
        for x in inputs:
            if x.ndim != 2:
                raise ValueError(self.name + ": currently Concat layer only support 2D inputs. Here x is" + str(x.ndim) +"D tensor." )
        return [T.concatenate(inputs, axis = 1)]


class Dropout(Layer):
    def __init__(self, p):
        self.name = self.__class__.__name__
        self.train_flag = True
        self.p = p
        self.srng = RandomStreams(seed=np.random.randint(920927))

    def f(self, inputs, is_train):
        out = []
        if isinstance(self.p, list):
            if len(self.p) != len(inputs):
                raise ValueError(self.name + ': The number of inputs should equal to the number of p if p is list.')

        for i in xrange(len(inputs)):
            x = inputs[i]
            p = self.p[i] if isinstance(self.p, list) else self.p 
            if self.p > 0.:
                retain_p = 1. - p
                if is_train:
                    x *= self.srng.binomial(x.shape, p=retain_p, dtype=theano.config.floatX) 
                else:
                    x *= retain_p 
                out.append(x)
            else:
                out.append(x)
        return out 

class Transformer(Layer):
    def __init__(self, regularizers = None, constraints = None):
        self.name = self.__class__.__name__ 

    def f(self, inputs):
        raise NotImplementedError(self.name)

# Below are different transforms
class Abs_diff(Transformer):
    def __init__(self, regularizers = None, constraints = None):
        self.name = self.__class__.__name__
        super(Abs_diff, self).__init__(regularizers, constraints)

    def f(self, inputs):
        if len(inputs) != 2:
            raise ValueError('the number of inputs for "abs_diff" transform should be 2.')
        return [abs(inputs[0] - inputs[1])]

class Fancy_diff(Transformer):
     def __init__(self, regularizers = None, constraints = None):
         self.name = self.__class__.__name__
         super(Fancy_diff, self).__init__(regularizers, constraints)

     def f(self, inputs):
         raise NotImplementedError()


class CostFunc(object):
    @staticmethod
    def nll(x, p, y):
        p_ = CostFunc.check_single(p)[0]
        y_ = CostFunc.check_single(y)[0]
        return -T.mean(T.log(p_)[T.arange(y_.shape[0]), y_])

    @staticmethod
    def check_single(v):
        if isinstance(v, list):
            if len(v)>1:
                raise ValueError('CostFunc.nll: len(v) should not exceed 1.')
        return v

class Model(object):
    def __init__(self):
        self.name = self.__class__.__name__
        self.layers = []
        self.params = []
        self.regularizers = []
        self.constraints = []
 
    def add(self, layer):
        self.layers.append(layer)
        self.params += layer.params if hasattr(layer, 'params') else []
        self.constraints += layer.constraints if hasattr(layer, 'constraints') else []
        self.regularizers += layer.regularizers if hasattr(layer, 'regularizers') else []

    def f(self, X, is_train):
        out = X 
        for layer in self.layers:
            out = layer.f(out, is_train) if hasattr(layer, 'train_flag') else layer.f(out)
        return out

    def build(self, cost_func, optimizer, X = [], Y = []):
        self.X = X if isinstance(X, list) else [X]
        self.Y = Y if isinstance(Y, list) else [Y]

        def build_f(is_train):
            out = self.X 
            for layer in self.layers:
                out = layer.f(out, is_train) if hasattr(layer, 'train_flag') else layer.f(out)
            cost = cost_func(None, out, self.Y)
            for r in self.regularizers:
                 cost = r(cost)
            return out, cost 

        self.out_train, self.cost_train = build_f(is_train = True)
        self.out_test, self.cost_test = build_f(is_train = False)

        self.optimizer = optimizer
        self.updates = self.optimizer.get_updates(self.params, self.constraints, self.cost_train)

        self.train = theano.function(self.X + self.Y, self.cost_train, updates = self.updates)
        self.test = theano.function(self.X + self.Y, [self.cost_test] + self.out_test)
        self.out = theano.function(self.X, self.out_test)

    def save(self, path_out):
        import pickle
        import sys
        sys.setrecursionlimit(10000)

        file_save = open(path_out, 'wb')
        pickle.dump(self, file_save)

    @staticmethod
    def load(path_in):
        import pickle
        import sys
        sys.setrecursionlimit(10000)

        file_load = open(path_in, 'rb') 
        m = pickle.load(file_load)
        return m
