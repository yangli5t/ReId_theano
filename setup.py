__author__ = 'austin'
import numpy


# All indexes here are ZERO based!
def get_abs_ind_per_cam_person(img_num_per_cam_person, person_inds, cam_inds):
    abs_person_cam_inds = numpy.repeat(person_inds * img_num_per_cam_person.shape[1], cam_inds.size)
    abs_person_cam_inds += numpy.tile(cam_inds, person_inds.size)
    cum_num_per_cam_person = numpy.cumsum(img_num_per_cam_person.flatten())
    cum_num_per_cam_person = numpy.append(0, cum_num_per_cam_person)

    abs_inds = numpy.asarray([], dtype='int32')
    # cam_label = numpy.asarray([], dtype='int32')
    # person_label = numpy.asarray([], dtype='int32')
    for ind in abs_person_cam_inds:
        abs_inds = numpy.append(abs_inds, numpy.arange(cum_num_per_cam_person[ind], cum_num_per_cam_person[ind+1]))
    return abs_inds
        # p_ind = ind / cam_inds.size
        # c_ind = ind % cam_inds.size
        # num = img_num_per_cam_person[p_ind][c_ind]
        # cam_label = numpy.append(cam_label, c_ind * numpy.ones((num,), dtype='int32'))
        # person_label = numpy.append(person_label, p_ind * numpy.ones((num,), dtype='int32'))

    # assert(abs_inds.shape == person_label.shape == cam_label.shape)
    # return [abs_inds, person_label, cam_label]


def get_abs_ind_per_person(img_num_per_cam_person, person_inds):
    cam_inds = numpy.arange(0, img_num_per_cam_person.shape[1])
    return get_abs_ind_per_cam_person(img_num_per_cam_person, person_inds, cam_inds)


def get_abs_ind_per_cam(img_num_per_cam_person, cam_inds):
    person_inds = numpy.arange(0, img_num_per_cam_person.shape[0])
    return get_abs_ind_per_cam_person(img_num_per_cam_person, person_inds, cam_inds)


def get_abs_ind_one(img_num_per_cam_person, p_ind, c_ind):
    ind = p_ind * img_num_per_cam_person.shape[1] + c_ind
    cum_img_num = numpy.append(0, numpy.cumsum(img_num_per_cam_person.flatten()))

    return numpy.arange(cum_img_num[ind], cum_img_num[ind+1])


def generate_ind_pairs(img_num_per_cam_person, person_inds, cam_inds):
    positive_pair_a = numpy.asarray([], dtype='int32')
    positive_pair_b = numpy.asarray([], dtype='int32')
    negative_pair_a = numpy.asarray([], dtype='int32')
    negative_pair_b = numpy.asarray([], dtype='int32')

    past_person_inds = numpy.asarray([], dtype='int32')
    for i in person_inds:
        past_person_inds = numpy.append(past_person_inds, i)
        past_cam_inds = numpy.asarray([], dtype='int32')
        for j in cam_inds:
            past_cam_inds = numpy.append(past_cam_inds, j)
            target_abs_inds = get_abs_ind_one(img_num_per_cam_person, i, j)
            pos_cand_abs_inds = get_abs_ind_per_cam_person(img_num_per_cam_person,
                                                           numpy.asarray([i], dtype='int32'),
                                                           numpy.setdiff1d(cam_inds, past_cam_inds))
            neg_cand_abs_inds = get_abs_ind_per_cam_person(img_num_per_cam_person,
                                                       numpy.setdiff1d(person_inds, past_person_inds),
                                                       numpy.setdiff1d(cam_inds, j))

            [tmp_a, tmp_b] = numpy.meshgrid(pos_cand_abs_inds, target_abs_inds)
            positive_pair_a = numpy.append(positive_pair_a, tmp_b.flatten())
            positive_pair_b = numpy.append(positive_pair_b, tmp_a.flatten())
            [tmp_a, tmp_b] = numpy.meshgrid(neg_cand_abs_inds, target_abs_inds)
            negative_pair_a = numpy.append(negative_pair_a, tmp_b.flatten())
            negative_pair_b = numpy.append(negative_pair_b, tmp_a.flatten())

    return [positive_pair_a, positive_pair_b, negative_pair_a, negative_pair_b]


def generate_batch_index(img_num_per_cam_person, cam_inds, batch_size):
    sub_img_num_per_cam_person = img_num_per_cam_person[:, cam_inds]
    available_cam_per_person = numpy.sum(sub_img_num_per_cam_person > 0, axis=1)
    tmp_inds = numpy.nonzero(available_cam_per_person > 1)[0]
    assert(tmp_inds.size < batch_size)
    return numpy.random.choice(tmp_inds, batch_size, replace=False)


def testing(test_model, img_num_per_cam_person, img_data, test_ind, test_size):
    rank = numpy.zeros((test_size,))
    for j in test_ind:
        target_inds = get_abs_ind_one(img_num_per_cam_person, j, 0)
        cand_inds = get_abs_ind_per_cam_person(
            img_num_per_cam_person,
            test_ind,
            numpy.asarray([1], dtype='int32')
        )

        [inds_a, inds_b] = numpy.meshgrid(target_inds, cand_inds)
        scores = test_model(img_data[inds_a.flatten(), :, :, :], img_data[inds_b.flatten(), :, :, :])[:, 0]

        test_num = img_num_per_cam_person[test_ind, 1] * img_num_per_cam_person[j, 0]
        test_cum_num = numpy.cumsum(test_num)
        assert(scores.size == test_cum_num[-1])
        score_per_person = numpy.array_split(scores, test_cum_num[:-1])
        score_sum_per_person = []
        for row in score_per_person:
            score_sum_per_person.append(numpy.sum(row))
        # score_sum_per_person = numpy.zeros((test_size,), dtype='float32')
        # cnt = 0
        # for p in test_ind:
        #     cand_inds = get_abs_ind_one(img_num_per_cam_person, p, 1)
        #     [inds_a, inds_b] = numpy.meshgrid(target_inds, cand_inds)
        #     scores = test_model(img_data[inds_a.flatten(), :, :, :], img_data[inds_b.flatten(), :, :, :])[:, 0]
        #     score_sum_per_person[cnt] = (numpy.sum(scores))
        #     cnt += 1
        score_sum_per_person = numpy.asarray(score_sum_per_person, dtype='float32')
        k = numpy.argwhere(test_ind == j)
        r = (test_size - 1) - numpy.argwhere(numpy.argsort(score_sum_per_person) == k[0])[0]
        rank[r] += 1

    rank = numpy.cumsum(rank)
    return rank

#
# total_size = 10
# train_size = 4
# test_size = total_size - train_size
# # img_num_per_cam_person = numpy.ones([1, total_size], dtype='int32')
# PN_ratio = 10
#
# train_ind = numpy.sort(numpy.random.choice(total_size, train_size, replace=False))
# test_ind = numpy.setdiff1d(range(total_size), train_ind)
#
# cam_ind = numpy.asarray([0, 1, 4], dtype='int32')
#
# img_num_per_cam_person = numpy.random.randint(0, 10, (10, 6))
#
# generate_batch_index(img_num_per_cam_person, cam_ind, train_size)
#
# generate_ind_pairs(img_num_per_cam_person, train_ind, cam_ind)
#
# for i in test_ind:
#     target_inds = get_abs_ind_one(img_num_per_cam_person, i, 0)
#     cand_inds = get_abs_ind_per_cam_person(img_num_per_cam_person, test_ind, 1)
#
#     [inds_a, inds_b] = numpy.meshgrid(target_inds, cand_inds)
#     scores = test_model(inds_a, inds_b)
#
#     test_num = img_num_per_cam_person[test_ind, 1] * img_num_per_cam_person[i, 0]
#     test_cum_num = numpy.cumsum(test_num)
#     score_per_person = numpy.array_split(scores, test_cum_num[:-1])
#     score_sum_per_person = []
#     for row in score_per_person:
#         score_sum_per_person.append(numpy.sum(row))
