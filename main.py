__author__ = 'austin'
import numpy
import theano
from theano import tensor as T
import Layer
from PIL import Image


def read_image(address):
    img = Image.open(open(address))
    img = numpy.asarray(img, dtype='float64') / 256.
    # put image in 4D tensor of shape (1, 3, height, width)
    img = img.transpose(2, 0, 1).reshape(1, 3, 128, 48)
    return img

img1 = read_image('/home/austin/Documents/Datasets/VIPeR/cam_a/001_45.bmp')
img2 = read_image('/home/austin/Documents/Datasets/VIPeR/cam_b/091_90.bmp')

rng = numpy.random.RandomState(23455)

X1 = T.tensor4('X1')
X2 = T.tensor4('X2')
Y = T.ivector('Y')

layer0 = Layer.ConvMaxPool2Layer(
    rng,
    input1=X1,
    input2=X2,
    image_shape=[1, 3, 128, 48],
    filter_shape=[20, 3, 5, 5],
    poolsize=[2, 2]
)

layer1 = Layer.ConvMaxPool2Layer(
    rng,
    input1=layer0.output1,
    input2=layer0.output2,
    image_shape=[1, 20, 62, 22],
    filter_shape=[25, 20, 5, 5],
    poolsize=[2, 2]
)

layer2_a = Layer.CrossInputNeighborDiffLayer(layer1.output1, layer1.output2)
layer2_b = Layer.CrossInputNeighborDiffLayer(layer1.output2, layer1.output1)

layer3_a = Layer.PatchSummaryLayer(rng, input=layer2_a.output, filter_shape=[25, 25, 5, 5], stride=5)
layer3_b = Layer.PatchSummaryLayer(rng, input=layer2_b.output, filter_shape=[25, 25, 5, 5], stride=5)

layer4_a = Layer.ConvMaxPoolLayer(
    rng,
    input=layer3_a.output,
    image_shape=[1, 25, 25, 5],
    filter_shape=[25, 25, 3, 3],
    poolsize=[2, 2]
)
layer4_b = Layer.ConvMaxPoolLayer(
    rng,
    input=layer3_b.output,
    image_shape=[1, 25, 25, 5],
    filter_shape=[25, 25, 3, 3],
    poolsize=[2, 2]
)

layer5 = Layer.HiddenLayer(rng, layer4_a.output, layer4_b.output, 11*1*50, 500)

layer6 = Layer.LogisticRegression(layer5.output, 500, 2)
cost = layer6.negative_log_likelihood(Y)

params = layer6.params + layer5.params + layer4_a.params + layer4_b.params + layer3_a.params + layer3_b.params + layer1.params + layer0.params
grads = T.grad(cost, params)
# train_model is a function that updates the model parameters by
# SGD Since this model has many parameters, it would be tedious to
# manually create an update rule for each model parameter. We thus
# create the updates list by automatically looping over all
# (params[i], grads[i]) pairs.
learning_rate = 0.01
updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
]


def read_image(address):
    img = Image.open(open(address))
    img = numpy.asarray(img, dtype='float32') / 256.
    # put image in 4D tensor of shape (1, 3, height, width)
    img = img.transpose(2, 0, 1).reshape(1, 3, 128, 48)
    return img


directory = '/home/austin/Documents/Datasets/VIPeR'
imgfiles1 = sorted(glob.glob(directory + '/cam_a/*.bmp'))
imgfiles2 = sorted(glob.glob(directory + '/cam_b/*.bmp'))

img_data_a = numpy.empty([0, 3, 128, 48], dtype='float32')
for img_address in imgfiles1:
    img = read_image(img_address)
    img_data_a = numpy.vstack([img_data_a, img])
img_data_b = numpy.empty([0, 3, 128, 48], dtype='float32')
for img_address in imgfiles2:
    img = read_image(img_address)
    img_data_b = numpy.vstack([img_data_b, img])

data_cam_a = theano.shared(img_data_a)
data_cam_b = theano.shared(img_data_b)


N = 632 # number of person in the dataset

train_size = 316
test_size = N - train_size
img_num_per_person = numpy.ones([1, N], dtype='int32')
PN_ratio = 10

train_ind = numpy.sort(numpy.random.choice(N, train_size, replace=False))
test_ind = numpy.setdiff1d(range(N), train_ind)

train_index_a = []
train_index_b = []
train_y = []
batch_person_num = 10
for i in range(200):
    target_ind = numpy.random.choice(train_ind, batch_person_num, replace=True)

    batch_train_index_a = numpy.empty([0,], dtype='int32')
    batch_train_index_b = numpy.empty([0,], dtype='int32')
    batch_train_y = numpy.empty([0,], dtype='int32')

    for ind in target_ind:
        candidate_ind = numpy.random.choice(train_ind, PN_ratio * 1, replace=False)
        while numpy.count_nonzero(candidate_ind == ind):
            candidate_ind = numpy.random.choice(train_ind, PN_ratio * 1, replace=False)

        tmp_index_a = numpy.repeat(ind, (PN_ratio + 1) * 1)
        tmp_index_b = numpy.hstack((ind, candidate_ind))
        batch_train_index_a = numpy.hstack((batch_train_index_a, tmp_index_a))
        batch_train_index_b = numpy.hstack((batch_train_index_b, tmp_index_b))
        tmp_y = (tmp_index_a == tmp_index_b) * 1
        batch_train_y = numpy.hstack((batch_train_y, tmp_y))

    train_index_a.append(batch_train_index_a)
    train_index_b.append(batch_train_index_b)
    train_y.append(batch_train_y)


index_cam_a = T.imatrix()
index_cam_b = T.imatrix()
y_value = T.imatrix()

train_model = theano.function(
    [index_cam_a, index_cam_b, y_value],
    cost,
    updates=updates,
    givens={
        X1: data_cam_a[index_cam_a],
        X2: data_cam_b[index_cam_b],
        Y: y_value
    }
)

