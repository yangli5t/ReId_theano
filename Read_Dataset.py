__author__ = 'austin'

import numpy
from PIL import Image
import glob

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
        tmp_y = (tmp_index_a == tmp_index_b) * 2 -1
        batch_train_y = numpy.hstack((batch_train_y, tmp_y))

    train_index_a.append(batch_train_index_a)
    train_index_b.append(batch_train_index_b)
    train_y.append(batch_train_y)


img_num_per_cam_person = numpy.random.randint(1, 100, (100, 2))


