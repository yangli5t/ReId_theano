__author__ = 'austin'
import numpy
import theano
from theano import tensor as T
import Layer
from PIL import Image
import glob
from keras.optimizers import SGD, RMSprop
from keras.constraints import identity

rng = numpy.random.RandomState(23455)

X1 = T.tensor4('X1', dtype='float32')
X2 = T.tensor4('X2', dtype='float32')
Y = T.ivector('Y')

layer0 = Layer.ConvMaxPool2Layer(
    rng,
    input1=X1,
    input2=X2,
    filter_shape=[25, 3, 3, 3],
    poolsize=[2, 2]
)

layer1 = Layer.ConvMaxPool2Layer(
    rng,
    input1=layer0.output1,
    input2=layer0.output2,
    filter_shape=[25, 25, 3, 3],
    poolsize=[2, 2]
)

layer2 = Layer.ConvMaxPoolLayer(
    rng,
    input=T.abs_(layer1.output1 - layer1.output2),
    filter_shape=[25, 25, 3, 3],
    poolsize=[2, 2]
)

layer3 = Layer.HiddenLayer(
    rng,
    input=layer2.output,
    n_in=25*14*4,
    n_out=500
)

layer4 = Layer.LogisticRegression(layer3.output, 500, 2)
cost = layer4.negative_log_likelihood(Y)

params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
grads = T.grad(cost, params)

# learning_rate = numpy.float32(0.01)
# updates = [
#     (param_i, param_i - learning_rate * grad_i)
#     for param_i, grad_i in zip(params, grads)
# ]

constraints_list = []
for param in params:
    constraints_list.append(identity())

rms = RMSprop()
updates = rms.get_updates(params, constraints_list, cost)

def read_image(address):
    img = Image.open(open(address))
    img = numpy.asarray(img, dtype='float32') / 256.
    # put image in 4D tensor of shape (1, 3, height, width)
    img = img.transpose(2, 0, 1).reshape(1, 3, 128, 48)
    return img


# img1 = read_image('/home/austin/Documents/Datasets/VIPeR/cam_a/001_45.bmp')
# img2 = read_image('/home/austin/Documents/Datasets/VIPeR/cam_b/091_90.bmp')
# f = theano.function([X1, X2, Y], [cost, layer2.similarity])
# y = numpy.asarray([-1], dtype='int32')
# [tmp, sim] = f(img1, img2, y)

# img1 = numpy.ones((2, 3, 128, 48), dtype='float32')
# img2 = numpy.ones((2, 3, 128, 48), dtype='float32')
# y = numpy.ones((2,), dtype='int32')
# f = theano.function([X1, X2, Y], cost)
# tmp = f(img1, img2, y)


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
PN_ratio = 5

train_ind = numpy.sort(numpy.random.choice(N, train_size, replace=False))
test_ind = numpy.setdiff1d(range(N), train_ind)

train_index_a = []#numpy.empty([0,], dtype='int32')
train_index_b = []#numpy.empty([0,], dtype='int32')
train_y = []#numpy.empty([0,], dtype='int32')

batch_person_num = 30
iterations = 5000
for i in range(iterations):
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
        tmp_y = 1 - ((tmp_index_a == tmp_index_b) * 1)
        batch_train_y = numpy.hstack((batch_train_y, tmp_y))

    # train_index_a = numpy.vstack((train_index_a, batch_train_index_a))
    # train_index_b = numpy.vstack((train_index_b, batch_train_index_b))
    train_index_a.append(batch_train_index_a)
    train_index_b.append(batch_train_index_b)
    train_y.append(batch_train_y)
    # train_y = numpy.vstack((train_y, batch_train_y))


train_index_a = numpy.int32(numpy.vstack(train_index_a))
train_index_b = numpy.int32(numpy.vstack(train_index_b))
train_y = numpy.int32(numpy.vstack(train_y))

test_index_a = numpy.int32(numpy.tile(numpy.reshape(test_ind, (test_size, 1)), (1, test_size)))
test_index_b = numpy.int32(numpy.tile(test_ind, (test_size, 1)))

label_y = theano.shared(train_y.flatten())
index_cam_a = theano.shared(train_index_a.flatten())
index_cam_b = theano.shared(train_index_b.flatten())
batch_size = theano.shared(numpy.int32(batch_person_num * (PN_ratio + 1)))

# index_cam_a = T.imatrix()
# index_cam_b = T.imatrix()
# index_y = T.imatrix()
index = T.iscalar()

train_model = theano.function(
    [index],
    cost,
    updates=updates,
    givens={
        X1: data_cam_a[index_cam_a[index * batch_size: (index + 1) * batch_size]],
        X2: data_cam_b[index_cam_b[index * batch_size: (index + 1) * batch_size]],
        Y: label_y[index * batch_size: (index + 1) * batch_size]
    }
)

test_index_cam_a = theano.shared(test_index_a.flatten())
test_index_cam_b = theano.shared(test_index_b.flatten())
test_batch_size = theano.shared(numpy.int32(test_size))
test_model = theano.function(
    [index],
    layer4.p_y_given_x,
    givens={
        X1: data_cam_a[test_index_cam_a[index * test_batch_size: (index + 1) * test_batch_size]],
        X2: data_cam_b[test_index_cam_b[index * test_batch_size: (index + 1) * test_batch_size]],
    }
)

for i in range(iterations):
    cost = train_model(i)
    if i % 100 == 0:
        rank = numpy.zeros((test_size, 1))
        for j in range(test_size):
            scores = test_model(j)[:, 0]
            k = numpy.argwhere(numpy.equal(test_index_a[j], test_index_b[j]))
            r = (test_size - 1) - numpy.argwhere(numpy.argsort(scores) == k[0])[0]
            rank[r] += 1
        rank = numpy.cumsum(rank)
        print('rank 1: %4.2f,   rank 5: %4.2f' % (rank[0]/test_size * 100, rank[4]/test_size * 100))

    if i % 10 == 0:
        print 'iteration: %d, cost: %f' % (i, cost)

