__author__ = 'austin'

import theano
import numpy
import scipy.io
import myModel
import cPickle as pickle
import setup


model = myModel.ReIdModel()
img_num_per_cam_person = pickle.load(open('cuhk03_img_num.p', 'rb'))
img_data = numpy.memmap('cuhk03_np.dat', dtype='float32', mode='r+', shape=(14096, 3, 160, 60))

train_model = theano.function(
    [model.X1, model.X2, model.Y],
    model.cost,
    updates=model.updates
)

test_model = theano.function(
    [model.X1, model.X2],
    model.layer5.p_y_given_x
)


total_size = img_num_per_cam_person.shape[0]
train_size = 1160
validation_size = 100
test_size = 100
PN_ratio = 1

all_ind = range(total_size)
all_ind = numpy.delete(all_ind, 1350)
assert(not all_ind.__contains__(1350))
train_ind = numpy.sort(numpy.random.choice(all_ind, train_size, replace=False))
non_train_ind = numpy.setdiff1d(all_ind, train_ind)
assert(non_train_ind.size == total_size - train_size - 1)
test_ind = numpy.sort(numpy.random.choice(non_train_ind, test_size, replace=False))
non_train_test_ind = numpy.setdiff1d(non_train_ind, test_ind)
validation_ind = numpy.sort(numpy.random.choice(non_train_test_ind, validation_size, replace=False))
assert(numpy.argwhere(numpy.equal(test_ind, validation_ind)).size == 0)
cam_ind = numpy.asarray([0, 1], dtype='int32')

batch_train_size = 5
iterations = 200001
for i in range(iterations):
    person_inds = numpy.sort(numpy.random.choice(train_ind, batch_train_size, replace=False))
    [positive_pair_a, positive_pair_b, negative_pair_a, negative_pair_b] = \
        setup.generate_ind_pairs(img_num_per_cam_person, person_inds, cam_ind)

    pos_n = positive_pair_a.size
    neg_n = negative_pair_a.size
    neg_n = min(pos_n * PN_ratio, neg_n)
    neg_ind = numpy.sort(numpy.random.choice(negative_pair_a.size, neg_n, replace=False))
    pair_ind_a = numpy.hstack((positive_pair_a, negative_pair_a[neg_ind]))
    pair_ind_b = numpy.hstack((positive_pair_b, negative_pair_b[neg_ind]))
    y = numpy.hstack((numpy.zeros((pos_n,), dtype='int32'), numpy.ones((neg_n,), dtype='int32')))

    cost = train_model(img_data[pair_ind_a, :, :, :], img_data[pair_ind_b, :, :, :], y)
    if i % 10 == 0:
        print 'iteration: %d, cost: %f' % (i, cost)

    if i % 1000 == 999:
        t_rank = setup.testing(test_model, img_num_per_cam_person, img_data, test_ind, test_size)
        print('test set: rank 1: %4.2f,   rank 5: %4.2f,   rank 10: %4.2f' %
              (t_rank[0]/test_size * 100, t_rank[4]/test_size * 100, t_rank[9]/test_size * 100))
        v_rank = setup.testing(test_model, img_num_per_cam_person, img_data, validation_ind, validation_size)
        print('validation set: rank 1: %4.2f,   rank 5: %4.2f,   rank 10: %4.2f' %
              (v_rank[0]/validation_size * 100, v_rank[4]/validation_size * 100, v_rank[9]/validation_size * 100))

        model.save_model(i)
        rank_mat = 'rank_iter_%i' % i + '.mat'
        scipy.io.savemat(rank_mat, {'v_rank': v_rank, 't_rank': t_rank})

