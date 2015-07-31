__author__ = 'austin'
import numpy
from PIL import Image
import glob
import os
import theano
from theano import tensor as T
from theano.tensor.nnet import neighbours
import Layer

#
# # Defining variables
# images = T.tensor4('images')
# neibs = neighbours.images2neibs(images, neib_shape=(5, 5),)
# # Constructing theano function
# window_function = theano.function([images], T.grad(T.sum(neibs), images))
#
# # Input tensor (one image 10x10)
# im_val = numpy.arange(100.).reshape((1, 1, 10, 10))
# # Function application
# neibs_val = window_function(im_val)
#
#
# # read images
def read_image(address):
    img = Image.open(open(address))
    img = numpy.asarray(img, dtype='float32') / 256.
    # put image in 4D tensor of shape (1, 3, height, width)
    img = img.transpose(2, 0, 1).reshape(1, 3, 160, 60)
    return img

#
# directory = '/home/austin/Documents/Datasets/VIPeR'
# imgfiles1 = sorted(glob.glob(directory + '/cam_a/*.bmp'))
# imgfiles2 = sorted(glob.glob(directory + '/cam_b/*.bmp'))
#
# img_data_a = numpy.empty([0, 3, 128, 48], dtype='float32')
# for img_address in imgfiles1:
#     img = read_image(img_address)
#     img_data_a = numpy.vstack([img_data_a, img])
# img_data_b = numpy.empty([0, 3, 128, 48], dtype='float32')
# for img_address in imgfiles2:
#     img = read_image(img_address)
#     img_data_b = numpy.vstack([img_data_b, img])




# # generate train/test indexes

# N = 632 # number of person in the dataset
#
# train_size = 316
# test_size = N - train_size
# img_num_per_person = numpy.ones([1, N], dtype='int32')
# PN_ratio = 10
#
# train_ind = numpy.sort(numpy.random.choice(N, train_size, replace=False))
# test_ind = numpy.setdiff1d(range(N), train_ind)
# test_index_a = numpy.int32(numpy.tile(numpy.reshape(test_ind, (test_size, 1)), (1, test_size)))
# test_index_b = numpy.int32(numpy.tile(test_ind, (test_size, 1)))
#
# k = numpy.argwhere(numpy.equal(test_index_a[4], test_index_b[4]))
# tmp = 315 - numpy.argwhere(numpy.argsort(test_ind) == k[0])[0]
#
# train_index_a = []
# train_index_b = []
# train_y = []
# batch_person_num = 10
# for i in range(200):
#     target_ind = numpy.random.choice(train_ind, batch_person_num, replace=True)
#
#     batch_train_index_a = numpy.empty([0,], dtype='int32')
#     batch_train_index_b = numpy.empty([0,], dtype='int32')
#     batch_train_y = numpy.empty([0,], dtype='int32')
#
#     for ind in target_ind:
#         candidate_ind = numpy.random.choice(train_ind, PN_ratio * 1, replace=False)
#         while numpy.count_nonzero(candidate_ind == ind):
#             candidate_ind = numpy.random.choice(train_ind, PN_ratio * 1, replace=False)
#
#         tmp_index_a = numpy.repeat(ind, (PN_ratio + 1) * 1)
#         tmp_index_b = numpy.hstack((ind, candidate_ind))
#         batch_train_index_a = numpy.hstack((batch_train_index_a, tmp_index_a))
#         batch_train_index_b = numpy.hstack((batch_train_index_b, tmp_index_b))
#         tmp_y = (tmp_index_a == tmp_index_b) * 2 - 1
#         batch_train_y = numpy.hstack((batch_train_y, tmp_y))
#
#     train_index_a.append(batch_train_index_a)
#     train_index_b.append(batch_train_index_b)
#     train_y.append(batch_train_y)

rng = numpy.random.RandomState(23455)
X1 = T.tensor4('X1', dtype='float32')
X2 = T.tensor4('X2', dtype='float32')
Y = T.ivector('Y')

layer0 = Layer.ConvMaxPool2Layer(
    rng,
    input1=X1,
    input2=X2,
    filter_shape=[25, 3, 5, 5],
    poolsize=[2, 2]
)

layer1 = Layer.ConvMaxPool2Layer(
    rng,
    input1=layer0.output1,
    input2=layer0.output2,
    filter_shape=[25, 25, 3, 3],
    poolsize=[2, 2]
)

layer2 = Layer.SecretLayer(
    rng,
    input1=layer1.output1,
    input2=layer1.output2,
    filter_shape=[25, 25, 5, 5]
)

# layer3 = Layer.MultiConvMaxPoolLayer(
#     rng,
#     input=layer2.results,
#     filter_shape=[25, 25, 3, 3],
#     poolsize=(2, 2)
# )

# layer3 = Layer.LocalCovLayerDropout(
#     rng,
#     input=layer2.results,
#     n_in=18*9*25,
#     n_out=200
# )
#
# layer4 = Layer.HiddenLayer(
#     rng,
#     input=layer3.output,
#     n_in=1600,
#     n_out=500
# )
#
test_model = theano.function(
    [X1, X2],
    layer2.results
)

img1 = read_image('/home/austin/Documents/Datasets/CUHK/cuhk03/cam_a/0001_01.png')
img2 = read_image('/home/austin/Documents/Datasets/CUHK/cuhk03/cam_b/0001_06.png')
output = test_model(img1, img2)
tmp = 1