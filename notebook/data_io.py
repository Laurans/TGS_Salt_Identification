from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
import cv2
from tqdm import tqdm
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import joblib

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i

class DataManager:

    def __init__(self):
        # Set some parameters
        self.im_width = 128
        self.im_height = 128
        self.im_chan = 3
        self.img_size_ori = 101

        self.path_train = '../data/train/'
        self.path_test = '../data/test/'

        self.train_ids = next(os.walk(self.path_train+"images"))[2]
        self.test_ids = next(os.walk(self.path_test+'images'))[2]

    def load_train(self):
        X_train = np.zeros((len(self.train_ids), self.im_height, self.im_width, self.im_chan), dtype=np.uint8)
        coverage = np.zeros((len(self.train_ids), 2), dtype=np.float32)
        Y_train = np.zeros((len(self.train_ids), self.im_height, self.im_width, 1), dtype=np.uint8)

        print('Getting and resizing train images and mask ...')

        for n, id_ in enumerate(tqdm(self.train_ids)):
            img = load_img(self.path_train + '/images/' + id_, color_mode = "grayscale")
            x = img_to_array(img)
            x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
            X_train[n] = np.dstack((x, x, x))

            mask = img_to_array(load_img(self.path_train + '/masks/' + id_, color_mode = "grayscale"))
            Y_train[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

            coverage[n, 0] = (mask / 255).sum() / self.img_size_ori**2
            coverage[n, 1] = cov_to_class(coverage[n, 0])

        print('Done!')

        return X_train, Y_train, coverage

    def load_test(self):
        X_test = np.zeros((len(self.test_ids), self.im_height, self.im_width, self.im_chan), dtype=np.uint8)

        sizes_test = []
        print('Getting and resizing test images ... ')

        for n, id_ in enumerate(tqdm(self.test_ids)):
            img = load_img(self.path_test + '/images/' + id_)
            x = img_to_array(img)[:,:,1]
            sizes_test.append([x.shape[0], x.shape[1]])
            x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
            X_test[n] = np.dstack((x, x, x))

        print('Done!')
        return X_test

    def downsample(self, list_img):
        def process_img(img):
            return resize(img, (self.img_size_ori, self.img_size_ori), mode='constant', preserve_range=True)

        return [np.squeeze(process_img(x)) for x in list_img]

    def save_dataset(self, obj):
        joblib.dump(obj, "../data/generated/dataset.bz2")

    def load_dataset(self):
        return joblib.load('../data/generated/dataset.bz2')


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

def get_predict_dict(preds_test_upsampled):
    return {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in enumerate(tqdm(test_ids))}
