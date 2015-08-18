__author__ = 'austin'

import numpy
import scipy.io
import theano
from theano import tensor as T
import Layer
from keras.optimizers import SGD, RMSprop
from keras.constraints import identity
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.utils.theano_utils import sharedX, floatX
from keras.regularizers import l1, l2
import pdb

class model_keras(object):
      def __init__(self):
          left = Sequential()
          left.add(Dense(784, 50))
          left.add(Activation('relu'))
          
          model = Sequential()
          model.add(Merge([left, left], mode='sum'))
          
          model.add(Dense(50, 10))
          model.add(Activation('softmax'))
          pdb.set_trace()

          model = Sequential()

          left = Sequential()
          num_kernel = 32
          l1_penalty = 0.0001
          b_mode = 'full'
          left.add(Convolution2D(num_kernel, 3, 2, 2,  W_regularizer=l1(l1_penalty), border_mode=b_mode))
          left.add(Convolution2D(num_kernel, num_kernel, 2, 2, W_regularizer=l1(l1_penalty), border_mode=b_mode))
          left.add(LeakyReLU(0.1))
          #left.add(Activation('relu'))
          left.add(MaxPooling2D(poolsize=(2, 2)))
          #left.add(Convolution2D(num_kernel, 3, 2, 2,  W_regularizer=l1(l1_penalty), border_mode=b_mode))
          #left.add(Convolution2D(num_kernel, num_kernel, 2, 2, W_regularizer=l1(l1_penalty), border_mode=b_mode))
          #left.add(LeakyReLU(0.1))
          ##left.add(Activation('relu'))
          #left.add(MaxPooling2D(poolsize=(2, 2)))

          model.add(Merge([left, left], mode='sum'))
          pdb.set_trace()
          self.f = theano.function(model.get_input(), model.get_output())


class ReIdModel(object):
    def __init__(self):
        rng = numpy.random.RandomState(23455)

        self.X1 = T.tensor4('X1', dtype='float32')
        self.X2 = T.tensor4('X2', dtype='float32')
        self.Y = T.ivector('Y')

        self.layer0 = Layer.ConvMaxPool2Layer(
            rng,
            input1=self.X1,
            input2=self.X2,
            filter_shape=[25, 3, 5, 5],
            poolsize=[2, 2]
        )

        self.layer1 = Layer.ConvMaxPool2Layer(
            rng,
            input1=self.layer0.output1,
            input2=self.layer0.output2,
            filter_shape=[25, 25, 3, 3],
            poolsize=[2, 2]
        )

        # self.layer2 = Layer.SecretLayer(
        #     rng,
        #     input1=self.layer1.output1,
        #     input2=self.layer1.output2,
        #     filter_shape=[25, 25, 5, 5]
        # )

        # self.layer3 = Layer.MultiConvMaxPoolLayer(
        #     rng,
        #     input=self.layer2.results,
        #     filter_shape=[25, 25, 3, 3],
        #     poolsize=(2, 2)
        # )
        #
        # self.layer3 = Layer.LocalCovLayerDropout(
        #     rng,
        #     input=self.layer2.results,
        #     n_in=18*9*25,
        #     n_out=200
        # )
        #
        # self.layer4 = Layer.HiddenLayerDropout(
        #     rng,
        #     train_input=self.layer3.train_output,
        #     test_input=self.layer3.test_output,
        #     # n_in=25*24*3,
        #     n_in=800,
        #     n_out=200
        # )
        self.layer2 = Layer.ConvMaxPoolLayer(
            rng,
            input=T.abs_(self.layer1.output1 - self.layer1.output2),
            filter_shape=[25, 25, 3, 3],
            poolsize=[2, 2]
        )

        self.layer3 = Layer.HiddenLayer(
            rng,
            input=self.layer2.output,
            n_in=25*18*5,
            n_out=500
        )

        self.layer5 = Layer.LogisticRegression(self.layer3.output, 500, 2)
        self.cost = self.layer5.negative_log_likelihood(self.Y)
        #
        # self.layer5 = Layer.LogisticRegressionDropout(
        #     train_input=self.layer4.train_output,
        #     test_input=self.layer4.test_output,
        #     n_in=200,
        #     n_out=2
        # )
        # self.cost = self.layer5.negative_log_likelihood_train(self.Y)

        self.params = self.layer5.params + self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        self.grads = T.grad(self.cost, self.params)

        # learning_rate = numpy.float32(0.01)
        # updates = [
        #     (param_i, param_i - learning_rate * grad_i)
        #     for param_i, grad_i in zip(params, grads)
        # ]

        constraints_list = []
        for param in self.params:
            constraints_list.append(identity())

        rms = RMSprop()
        self.updates = rms.get_updates(self.params, constraints_list, self.cost)

    def save_model(self, i):
        for model_index in range(len(self.params)):
            save_mat = 'model_%d' % model_index + '_iter_%i' % i + '.mat'
            scipy.io.savemat(save_mat, {'param': self.params[model_index].get_value()})
        return()
