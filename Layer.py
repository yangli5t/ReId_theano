__author__ = 'austin'
import numpy
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.tensor.nnet import neighbours
from theano.tensor.shared_randomstreams import RandomStreams


class ConvMaxPoolLayer(object):
    def __init__(self, rng, input, filter_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = theano.shared(
            value=numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
            borrow=True
        )

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.relu(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class ConvMaxPool2Layer(object):
    def __init__(self, rng, input1, input2, filter_shape, poolsize=(2, 2)):

        self.input1 = input1
        self.input2 = input2

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype='float32'
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = theano.shared(
            value=numpy.zeros((filter_shape[0],), dtype='float32'),
            borrow=True
        )

        # convolve input feature maps with filters
        conv_out1 = conv.conv2d(
            input=input1,
            filters=self.W,
            filter_shape=filter_shape
        )
        conv_out2 = conv.conv2d(
            input=input2,
            filters=self.W,
            filter_shape=filter_shape
        )
        # downsample each feature map individually, using maxpooling
        pooled_out1 = downsample.max_pool_2d(
            input=conv_out1,
            ds=poolsize,
            ignore_border=True
        )
        pooled_out2 = downsample.max_pool_2d(
            input=conv_out2,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output1 = T.nnet.relu(pooled_out1 + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output2 = T.nnet.relu(pooled_out2 + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input1 = input1
        self.input2 = input2


class MultiConvMaxPoolLayer(object):

    def __init__(self, rng, input, filter_shape, poolsize=(2, 2)):

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        self.W = []
        self.b = []
        self.output = []
        self.params = []

        for i in range(8):
            subinput = input[:, :, i*9:(i+1)*9, :]
            self.W.append(theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype='float32'
                ),
                borrow=True
            ))

            self.b.append(theano.shared(
                value=numpy.zeros((filter_shape[0],), dtype='float32'),
                borrow=True
            ))

            conv_out = conv.conv2d(
                input=subinput,
                filters=self.W[i],
                filter_shape=filter_shape
            )

            pooled_out = downsample.max_pool_2d(
                input=conv_out,
                ds=poolsize,
                ignore_border=True
            )

            self.output.append(T.nnet.relu(pooled_out + self.b[i].dimshuffle('x', 0, 'x', 'x')))
            self.params.append(self.W[i])
            self.params.append(self.b[i])

        self.output = T.concatenate(self.output, axis=2)


class CrossInputNeighborDiffLayer(object):
    def __init__(self, input1, input2):
        x1_sub = input1[:, :, 2:-2, 2:-2]
        x1_flatten = T.flatten(x1_sub)
        x1 = T.extra_ops.repeat(x1_flatten, 25)
        x1 = T.reshape(x1, [T.shape(x1_flatten)[0], 25])
        x2 = neighbours.images2neibs(input2, neib_shape=(5, 5), neib_step=(1, 1))
        diff = x1 - x2
        new_shape = T.shape(x1_sub)*[1, 1, 5, 5]
        diff_img = neighbours.neibs2images(diff, neib_shape=(5, 5), original_shape=[1, 25, 25*5, 5*5])
        self.output = T.nnet.relu(diff_img)


class PatchSummaryLayer(object):
    def __init__(self, rng, input, filter_shape, stride):

        self.input = input
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / (stride * stride))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        self.b = theano.shared(
            value=numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
            borrow=True
        )
        conv_out = conv.conv2d(input, self.W, subsample=(stride, stride))
        self.output = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is nnet.relu

        Hidden unit activation is given by: nnet.relu(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input.flatten(2)
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for nnet.relu activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to nnet.relu
        #        We have no info for other function, so we use the same as
        #        nnet.relu.

        # if activation == theano.tensor.nnet.sigmoid:
        #     W_values *= 4

        self.W = theano.shared(
            value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        lin_output = T.dot(self.input, self.W) + self.b

        self.output = T.nnet.relu(lin_output)
        self.params = [self.W, self.b]


class HiddenLayerDropout(object):
    def __init__(self, rng, train_input, test_input, n_in, n_out):

        # self.input = input.flatten(2)

        self.W = theano.shared(
            value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros((n_out,), dtype=theano.config.floatX),
            name='b',
            borrow=True
        )

        p = 0.5

        tmp_output = T.nnet.relu(T.dot(train_input.flatten(2), self.W) + self.b)
        srng = RandomStreams(rng.randint(1234))
        mask = (srng.uniform(size=tmp_output.shape) < p)/p

        self.train_output = tmp_output * mask
        self.test_output = T.nnet.relu(T.dot(test_input.flatten(2), self.W) + self.b)
        self.params = [self.W, self.b]


class Hidden2Layer(object):
    def __init__(self, rng, input1, input2, n_in, n_out):

        self.input1 = input1.flatten(2)
        self.input2 = input2.flatten(2)

        self.W = theano.shared(
            value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype='float32'
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=numpy.zeros((n_out,), dtype='float32'),
            name='b',
            borrow=True
        )

        lin_output1 = T.dot(self.input1, self.W) + self.b
        lin_output2 = T.dot(self.input2, self.W) + self.b

        self.output1 = T.nnet.relu(lin_output1)
        self.output2 = T.nnet.relu(lin_output2)
        self.similarity = self.similarity_func(self.output1, self.output2)
        self.params = [self.W, self.b]

    def similarity_func(self, x1, x2):
        return T.nnet.softmax(T.sum((x1 - x2) * (x1 - x2), axis=1))
        # return T.sum(x1*x2, axis=1) / (T.sqrt(T.sum(x1*x1, axis=1) * T.sum(x2*x2, axis=1)))

    def cost(self, y):
        return -T.mean(T.log(self.similarity)[T.arange(y.shape[0]), y])


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class LogisticRegressionDropout(object):
    def __init__(self, train_input, test_input, n_in, n_out):

        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        self.p_y_given_x_train = T.nnet.softmax(T.dot(train_input, self.W) + self.b)
        self.p_y_given_x_test = T.nnet.softmax(T.dot(test_input, self.W) + self.b)
        self.params = [self.W, self.b]

    def negative_log_likelihood_train(self, y):
        return -T.mean(T.log(self.p_y_given_x_train)[T.arange(y.shape[0]), y])

    def negative_log_likelihood_test(self, y):
        return -T.mean(T.log(self.p_y_given_x_test)[T.arange(y.shape[0]), y])



class SecretLayer(object):
    def __init__(self, rng, input1, input2, filter_shape):

        self.input1 = input1
        self.input2 = input2
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))

        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype='float32'
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        self.b = theano.shared(
            value=numpy.zeros((filter_shape[0],), dtype='float32'),
            borrow=True
        )

        self.block_i = theano.shared(
            value=numpy.asarray([0, 5, 10, 15, 18, 23, 28, 33])
        )

        n = theano.shared(value=numpy.int32(9*9*8))

        self.params = [self.W, self.b]

        results, updates = theano.scan(
            fn=self.patch_diff,
            outputs_info=None,
            sequences=T.arange(n),
            non_sequences=[input1, input2]
        )
        results = results.dimshuffle(1, 2, 0, 3, 4)
        self.results = results.reshape((results.shape[0], results.shape[1], 9*8, 9))

    def patch_diff(self, n, X1, X2):
        block_size = 5
        i = (n % (9 * 9)) / 9
        j = n % 9
        k = n / (9 * 9)

        sub_x1 = X1[:, :, self.block_i[k]:self.block_i[k]+block_size, i:i+block_size]
        sub_x2 = X2[:, :, self.block_i[k]:self.block_i[k]+block_size, j:j+block_size]
        # tmp = T.sqrt(T.sum(sub_x1 ** 2, axis=(2, 3)) * T.sum(sub_x2 ** 2, axis=(2, 3))).flatten()#.reshape((sub_x2.shape[0], sub_x2.shape[1], 1, 1))#.dimshuffle(0, 1, 'x', 'x') #
        # tmp = T.extra_ops.repeat(tmp, 25).reshape((sub_x1.shape[0], sub_x1.shape[1], 5, 5))
        conv_out = conv.conv2d(T.abs_(sub_x1 - sub_x2), self.W)
        #
        return T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))


# class LocalCovLayer(object):
#     def __init__(self, rng, input, n_in, n_out):
#         self.input1 = input[:, :, 0:18, :].flatten(2)
#         self.input2 = input[:, :, 18:36, :].flatten(2)
#         self.input3 = input[:, :, 36:54, :].flatten(2)
#         self.input4 = input[:, :, 54:72, :].flatten(2)
#
#         self.W1 = theano.shared(
#             value=numpy.asarray(
#                 rng.uniform(
#                     low=-numpy.sqrt(6. / (n_in + n_out)),
#                     high=numpy.sqrt(6. / (n_in + n_out)),
#                     size=(n_in, n_out)
#                 ),
#                 dtype=theano.config.floatX
#             ),
#             borrow=True
#         )
#
#         self.b1 = theano.shared(
#             value=numpy.zeros((n_out,), dtype=theano.config.floatX),
#             borrow=True
#         )
#         self.W2 = theano.shared(
#             value=numpy.asarray(
#                 rng.uniform(
#                     low=-numpy.sqrt(6. / (n_in + n_out)),
#                     high=numpy.sqrt(6. / (n_in + n_out)),
#                     size=(n_in, n_out)
#                 ),
#                 dtype=theano.config.floatX
#             ),
#             borrow=True
#         )
#
#         self.b2 = theano.shared(
#             value=numpy.zeros((n_out,), dtype=theano.config.floatX),
#             borrow=True
#         )
#         self.W3 = theano.shared(
#             value=numpy.asarray(
#                 rng.uniform(
#                     low=-numpy.sqrt(6. / (n_in + n_out)),
#                     high=numpy.sqrt(6. / (n_in + n_out)),
#                     size=(n_in, n_out)
#                 ),
#                 dtype=theano.config.floatX
#             ),
#             borrow=True
#         )
#
#         self.b3 = theano.shared(
#             value=numpy.zeros((n_out,), dtype=theano.config.floatX),
#             borrow=True
#         )
#         self.W4 = theano.shared(
#             value=numpy.asarray(
#                 rng.uniform(
#                     low=-numpy.sqrt(6. / (n_in + n_out)),
#                     high=numpy.sqrt(6. / (n_in + n_out)),
#                     size=(n_in, n_out)
#                 ),
#                 dtype=theano.config.floatX
#             ),
#             borrow=True
#         )
#
#         self.b4 = theano.shared(
#             value=numpy.zeros((n_out,), dtype=theano.config.floatX),
#             borrow=True
#         )
#
#         lin_output1 = T.dot(self.input1, self.W1) + self.b1
#         lin_output2 = T.dot(self.input2, self.W2) + self.b2
#         lin_output3 = T.dot(self.input3, self.W3) + self.b3
#         lin_output4 = T.dot(self.input4, self.W4) + self.b4
#
#         # self.output = T.nnet.relu(lin_output1)
#         self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
#         self.output = T.nnet.relu(T.concatenate([lin_output1, lin_output2, lin_output3, lin_output4], axis=1))


class LocalCovLayer(object):
    def __init__(self, rng, input, n_in, n_out):
        self.W = []
        self.b = []
        lin_output = []
        self.params = []

        for i in range(4):
            sub_input = input[:, :, i*18:(i+1)*18, :].flatten(2)
            self.W.append(
                theano.shared(
                    value=numpy.asarray(
                        rng.uniform(
                            low=-numpy.sqrt(6. / (n_in + n_out)),
                            high=numpy.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)
                        ),
                        dtype='float32'
                    ),
                    borrow=True
                )
            )

            self.b.append(
                theano.shared(
                    value=numpy.zeros((n_out,), dtype='float32'),
                    borrow=True
                )
            )
            self.params.append(self.W[i])
            self.params.append(self.b[i])

            lin_output.append(T.nnet.relu(T.dot(sub_input, self.W[i]) + self.b[i]))

        self.output = T.concatenate(lin_output, axis=1)


class LocalCovLayerDropout(object):
    def __init__(self, rng, input, n_in, n_out):
        self.W = []
        self.b = []
        train_output = []
        test_output = []
        self.params = []
        p = 0.5

        for i in range(4):
            sub_input = input[:, :, i*18:(i+1)*18, :].flatten(2)
            self.W.append(
                theano.shared(
                    value=numpy.asarray(
                        rng.uniform(
                            low=-numpy.sqrt(6. / (n_in + n_out)),
                            high=numpy.sqrt(6. / (n_in + n_out)),
                            size=(n_in, n_out)
                        ),
                        dtype='float32'
                    ),
                    borrow=True
                )
            )

            self.b.append(
                theano.shared(
                    value=numpy.zeros((n_out,), dtype='float32'),
                    borrow=True
                )
            )
            self.params.append(self.W[i])
            self.params.append(self.b[i])

            tmp_output = T.nnet.relu(T.dot(sub_input, self.W[i]) + self.b[i])
            srng = RandomStreams(rng.randint(1234))
            mask = (srng.uniform(size=tmp_output.shape) < p)/p
            train_output.append(tmp_output * mask)
            test_output.append(tmp_output)

        self.train_output = T.concatenate(train_output, axis=1)
        self.test_output = T.concatenate(test_output, axis=1)

