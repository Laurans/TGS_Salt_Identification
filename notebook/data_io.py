from skimage.io import imread, imshow, concatenate_images
from skimage.util import pad, crop
from skimage.transform import resize
from skimage import exposure

import cv2
from tqdm import tqdm
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import joblib
import pandas as pd

def cov_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i :
            return i

class DataManager:

    def __init__(self):
        # Set some parameters
        self.im_width = 124
        self.im_height = 124
        self.im_chan = 1
        self.img_size_ori = 101
        self.margin = [[11, 11+1], [11, 11+1]]

        self.path_train = '../data/train/'
        self.path_test = '../data/test/'
        self.depths = pd.read_csv('../data/depths.csv')

        self.train_ids = next(os.walk(self.path_train+"images"))[2]
        self.test_ids = next(os.walk(self.path_test+'images'))[2]
        
    def load_train(self):
        X_train = np.zeros((len(self.train_ids), self.im_height, self.im_width, self.im_chan), dtype=np.uint8)
        coverage = np.zeros((len(self.train_ids), 2), dtype=np.float32)
        Y_train = np.zeros((len(self.train_ids), self.im_height, self.im_width, self.im_chan ), dtype=np.uint8)
        train_depth = np.zeros((len(self.train_ids),))

        print('Getting and resizing train images and mask ...')

        for n, id_ in enumerate(tqdm(self.train_ids)):
            img = load_img(self.path_train + '/images/' + id_, color_mode = "grayscale")
            x = np.squeeze(img_to_array(img))
            x = np.expand_dims(pad(x, pad_width=self.margin, mode='symmetric'), -1)
            X_train[n] = x

            mask_ori = img_to_array(load_img(self.path_train + '/masks/' + id_, color_mode = "grayscale"))
            mask = np.squeeze(mask_ori)
            mask = np.expand_dims(pad(mask, pad_width=self.margin, mode='symmetric'), -1)
            Y_train[n] = mask

            coverage[n, 0] = (mask_ori / 255).sum() / self.img_size_ori**2
            coverage[n, 1] = cov_to_class(coverage[n, 0])

            train_depth[n] = self.depths[self.depths.id == id_.split('.')[0]]['z'].values[0]

        print('Done!')

        return X_train, Y_train, coverage, train_depth

    def load_test(self):
        X_test = np.zeros((len(self.test_ids), self.im_height, self.im_width, self.im_chan), dtype=np.uint8)
        test_depth = np.zeros((len(self.test_ids,)))
        print('Getting and resizing test images ... ')

        for n, id_ in enumerate(tqdm(self.test_ids)):
            img = load_img(self.path_test + '/images/' + id_, color_mode = "grayscale")
            x = np.squeeze(img_to_array(img))
            x = pad(x, pad_width=self.margin, mode='symmetric')
            X_test[n] = np.expand_dims(x, -1)

            test_depth[n] = self.depths[self.depths.id == id_.split('.')[0]]['z'].values[0]

        print('Done!')
        return X_test, test_depth

    def downsample(self, list_img):
        def process_img(img):
            return np.expand_dims(crop(np.squeeze(img), crop_width=self.margin), -1) #resize(img, (self.img_size_ori, self.img_size_ori), mode='constant', preserve_range=True)

        return np.array([np.squeeze(process_img(x)) for x in list_img])

    def save_dataset(self, obj, name='dataset'):
        joblib.dump(obj, "../data/generated/{}.bz2".format(name))

    def load_dataset(self, name='dataset'):
        return joblib.load('../data/generated/{}.bz2'.format(name))


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

def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(rle_mask):
    '''
    rle_mask: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(101*101, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(101,101)

def get_predict_dict(preds_test_upsampled):
    return {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in enumerate(tqdm(test_ids))}

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.color import gray2rgb
from skimage.color import rgb2gray

#Original_image = Image which has to labelled
#Mask image = Which has been labelled by some technique..
def crf(original_image, mask_img):

    # Converting annotated image to RGB if it is Gray scale
    #if(len(mask_img.shape)<3):
    mask_img = gray2rgb(np.squeeze(mask_img))

#     #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask_img[:,:,0] + (mask_img[:,:,1]<<8) + (mask_img[:,:,2]<<16)

#     # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2

    #Setting up the CRF model
    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

    #Run Inference for 10 steps
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((original_image.shape[0],original_image.shape[1]))
