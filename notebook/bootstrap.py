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
import pandas as pd

params = pd.read_csv('bootstrap_config.csv').as_matrix()

datamanager = DataManager()

X_train, Y_train, coverage = datamanager.load_train()

for i, (start, depth, residual, maxpool, upconv, aug_noise, aug_contrast, seed, vggunet, gaus_noise) in enumerate(params):
    print('-'*10)
    print('Start {}/{} model'.format(i, len(params)))
    print('-'*10)
    x_train, x_valid, y_train, y_valid, cov_train, cov_valid = train_test_split(
        X_train, Y_train, coverage, test_size=0.15, stratify=coverage[:, 1], random_state=seed)

    x_train_, y_train_ = augment_images(x_train, y_train, with_constrast=aug_contrast, with_noise=aug_noise)

    y_train_ = np.piecewise(y_train_, [y_train_ > 127.5, y_train_ < 127.5], [1, 0])

    y_valid = np.piecewise(y_valid, [y_valid > 127.5, y_valid < 127.5], [1, 0])

    amodel = create_model((datamanager.im_height, datamanager.im_width, datamanager.im_chan), vggunet, start, depth, maxpool, upconv, residual, gaus_noise)
    history = fit(amodel, x_train_, y_train_, x_valid, y_valid, 'model_{}.h5'.format(i))

    print('-'*10)
    print('End {} model'.format(i))
    print('-'*10)
