import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.random.seed(1)

import tensorflow as tf
tf.set_random_seed(2)

from data_io import *
from data_augmentation import *
from metrics import *
from model import *

from sklearn.model_selection import train_test_split

datamanager = DataManager()

X_train, Y_train, coverage = datamanager.load_train()

params = [(32, 5, False, True, 24), (32, 5, False, True, 42),
          (16, 6, False, True, 78), (16, 6, False, False, 87),
          (32, 5, False, False, 69), (32, 5, False, False, 96),
          (32, 5, False, True, 12), (32, 5, False, True, 21),
          (64, 4, False, False, 17), (32, 5, True, True, 71)]

for i, (start, depth, residual, maxpool, seed) in enumerate(params):
    print('-'*10)
    print('Start {} model'.format(i))
    print('-'*10)
    x_train, x_valid, y_train, y_valid, cov_train, cov_valid = train_test_split(
        X_train, Y_train, coverage, test_size=0.15, stratify=coverage[:, 1], random_state=seed)

    x_train_, y_train_ = augment_images(x_train, y_train)

    y_train_ = np.piecewise(y_train_, [y_train_ > 127.5, y_train_ < 127.5], [1, 0])

    y_valid = np.piecewise(y_valid, [y_valid > 127.5, y_valid < 127.5], [1, 0])

    amodel = UNet((datamanager.im_height, datamanager.im_width, datamanager.im_chan),
                  start_ch=start, depth=depth, batchnorm=True, residual=residual,
                  maxpool=maxpool, upconv=True)
    amodel.compile(optimizer='adam', loss=mixed_dice_bce_loss)
    history = fit(amodel, x_train_, y_train_, x_valid, y_valid, 'model_{}.h5'.format(i))


    print('-'*10)
    print('End {} model'.format(i))
    print('-'*10)
