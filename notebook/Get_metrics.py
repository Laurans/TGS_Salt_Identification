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
import joblib
from sklearn.model_selection import train_test_split
import datetime

datamanager = DataManager()

print('Loading dataset')
start_time = datetime.datetime.now()
#(x_train, y_train), (x_valid, y_valid) = datamanager.load_dataset()
X_train, Y_train, coverage = datamanager.load_train()
x_train, x_valid, y_train, y_valid, cov_train, cov_valid = train_test_split(
    X_train, Y_train, coverage, test_size=0.15, stratify=coverage[:, 1], random_state=12)

x_train_, y_train_ = augment_images(x_train, y_train)

y_train_ = np.piecewise(y_train_, [y_train_ > 127.5, y_train_ < 127.5], [1, 0])

y_valid = np.piecewise(y_valid, [y_valid > 127.5, y_valid < 127.5], [1, 0])

time_delta = datetime.datetime.now() - start_time
print('Loading time', time_delta)

#amodel = create_model(datamanager.im_height, datamanager.im_width, datamanager.im_chan)
#history = fit(amodel, x_train, y_train, x_valid, y_valid, 'model.h5')

model = load_model('model_6.h5', custom_objects={'mixed_dice_bce_loss': mixed_dice_bce_loss, 'dice_loss': dice_loss})
preds_valid = model.predict(x_valid, verbose=1)

thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds, desc='compute ious for thres')])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

print("Threshold ", threshold_best, 'Ious', iou_best)
