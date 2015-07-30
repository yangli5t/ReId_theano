__author__ = 'austin'

import numpy
from PIL import Image
import glob
import pickle


def read_image(address):
    img = Image.open(open(address))
    img = numpy.asarray(img, dtype='float32') / 256.
    # put image in 4D tensor of shape (1, 3, height, width)
    img = img.transpose(2, 0, 1).reshape(1, 3, 160, 60)
    return img

directory = '/home/austin/Documents/Datasets/CUHK/cuhk03'

img_num_per_cam_person = numpy.empty((0, 2), dtype='int32')
# img_data = numpy.empty([0, 3, 160, 60], dtype='float32')
img_data = numpy.memmap('cuhk03_np.dat', dtype='float32', mode='w+', shape=(14096, 3, 160, 60))

i = 0
for id in range(1, 1467+1):
    tmp = '%04d' % id
    imgfiles1 = sorted(glob.glob(directory + '/cam_a/' + tmp + '*.png'))
    imgfiles2 = sorted(glob.glob(directory + '/cam_b/' + tmp + '*.png'))

    img_num = numpy.asarray([len(imgfiles1), len(imgfiles2)], dtype='int32')
    if numpy.sum(img_num) == 0:
        continue
    img_num_per_cam_person = numpy.vstack((img_num_per_cam_person, img_num))

    for img_address in imgfiles1:
        img = read_image(img_address)
        # img_data = numpy.vstack([img_data, img])
        img_data[i, :, :, :] = img
        i += 1
    for img_address in imgfiles2:
        img = read_image(img_address)
        # img_data = numpy.vstack([img_data, img])
        img_data[i, :, :, :] = img
        i += 1
    if id%10 == 0:
        print id

img_data.flush()
pickle.dump(img_num_per_cam_person, open('cuhk03_img_num.p', 'wb'))