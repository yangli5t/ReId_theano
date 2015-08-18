__author__ = 'austin, Saizheng Zhang'

import theano
import numpy
import scipy.io
import myModel1
import cPickle as pickle
import setup
import time
import model_new
import os

import pdb
import numpy as np


img_num_per_cam_person = pickle.load(open('cuhk03_img_num.p', 'rb'))
img_data = numpy.memmap('cuhk03_np.dat', dtype='float32', mode='r+', shape=(14096, 3, 160, 60))
total_size = img_num_per_cam_person.shape[0]


def get_test_ind_list_cuhk03(path, use_labeled = True):
    import h5py
    import numpy as np
    f = h5py.File(path, 'r')

    labeled = f.get('labeled')
    detected = f.get('detected')
    reference = labeled if use_labeled else detected
 
    testsets = f.get('testsets')

    cell_sizes = [0] + [sum([f[reference[0, j]].shape[1] for j in xrange(i+1)]) for i in xrange(reference.shape[1])]
    test_ind_list = [] # testsets has 20 hdf5 object references
    for i in xrange(testsets.shape[1]):
        ind_temp = np.array(f[testsets[0, i]]).transpose().astype('int32')
        ind = np.array([(cell_sizes[ind_temp[k,0]-1] + ind_temp[k,1]-1) for k in xrange(ind_temp.shape[0])])
        test_ind_list.append(ind)
    return test_ind_list

 
def get_ind_list_zsz():
    all_ind = range(total_size)[:1360]
    special_ind = 1350
    test_ind_list = get_test_ind_list_cuhk03('./cuhk-03.mat')
    ind_list = []
    for i in xrange(len(test_ind_list)):
        test_ind = test_ind_list[i]
        non_test_ind = np.setdiff1d(np.setdiff1d(all_ind, test_ind), [special_ind])
        np.random.shuffle(non_test_ind)

        train_ind, validation_ind = np.split(non_test_ind, [1160])
        test_ind = np.setdiff1d(test_ind, [special_ind])

        ind_list.append([train_ind, validation_ind, test_ind])
    return ind_list


def train_model(index, mymodel, model_version):
    model = mymodel 
    train_model = model.train
    test_model = theano.function(model.X, model.f(model.X, is_train = True)[0])
    
    PN_ratio = 1

    ind_list = get_ind_list_zsz()
    train_ind, validation_ind, test_ind = ind_list[index] 
    train_size, validation_size ,test_size = train_ind.shape[0], validation_ind.shape[0], test_ind.shape[0]
    cam_ind = numpy.asarray([0, 1], dtype='int32')
    
    batch_train_size = 5 
    iterations = 200001
    rank_1 = 0
    rank_5 = 0
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
        # dangerous! Do not touch any code above after the "for"!
    
        cost = train_model(img_data[pair_ind_a, :, :, :], img_data[pair_ind_b, :, :, :], y)
    
        if i % 10 == 0:
            print 'iteration: %d, cost: %f' % (i, cost)
    
        if i % 500 == 499:
            t_start = time.time()
            v_rank = setup.testing(test_model, img_num_per_cam_person, img_data, validation_ind, validation_size)
            print('validation set: rank 1: %4.2f,   rank 5: %4.2f,   rank 10: %4.2f' %
                  (v_rank[0]/validation_size * 100, v_rank[4]/validation_size * 100, v_rank[9]/validation_size * 100))
            t_end = time.time()
            print "time consumed is " + str(t_end - t_start) + 'secs.'
    
            t_start = time.time()
            t_rank = setup.testing(test_model, img_num_per_cam_person, img_data, test_ind, test_size)
            print('test set: rank 1: %4.2f,   rank 5: %4.2f,   rank 10: %4.2f' %
                  (t_rank[0]/test_size * 100, t_rank[4]/test_size * 100, t_rank[9]/test_size * 100))
            t_end = time.time()
            print "time consumed is " + str(t_end - t_start) + 'secs.'
    
            folder_out = './models/' + 'model_' + model_version + '_' + str(index)
            if not os.path.isdir(folder_out):
                os.mkdir(folder_out)
    
            if v_rank[0]/validation_size *100 > rank_1 or v_rank[4]/validation_size *100 > rank_5:
               rank_1 = v_rank[0]/validation_size *100 
               rank_5 = v_rank[4]/validation_size *100
               rank_1_t = t_rank[0]/test_size*100
               rank_5_t = t_rank[4]/test_size*100
               model.save(folder_out + '/v_' + str(int(rank_1)) + '_' + str(int(rank_5)) \
                          + '_t_' + str(int(rank_1_t)) + '_' + str(int(rank_5_t)) + '.mdl')


if __name__ == "__main__":
    index_test = 0
    mymodel = model_new.V1_reg().model
    model_version = 'v1_reg'
    train_model(index_test, mymodel, model_version)
