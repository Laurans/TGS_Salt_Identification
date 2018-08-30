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
from losses import *
import joblib
from sklearn.model_selection import train_test_split
import datetime

TRAIN = False
CRF = True

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

assert x_train_.shape[1:] == x_valid.shape[1:]
time_delta = datetime.datetime.now() - start_time
print('Loading time', time_delta)

if TRAIN:
    amodel = create_model((datamanager.im_height, datamanager.im_width, datamanager.im_chan), 'unet', start_ch=32, depth=5)
    amodel.summary()
    history = fit(amodel, x_train_, y_train_, x_valid, y_valid, 'model.h5')

model = load_model('model.h5', custom_objects={'mixed_dice_bce_loss': mixed_dice_bce_loss, 'dice_loss': dice_loss, 'iou_metric':iou_metric, 'focal_loss':focal_loss})
tta_model = TTA_ModelWrapper(model)
preds_valid = tta_model.predict(x_valid)
iou_best, threshold_best = best_iou_and_threshold(y_valid, preds_valid, plot=True)
y_pred = np.int32(preds_valid > threshold_best)

print('plot_prediction')
x_valid = datamanager.downsample(x_valid[:,:,:, 0])
y_valid = datamanager.downsample(y_valid)
y_pred = datamanager.downsample(y_pred)
plot_prediction(x_valid, y_valid, y_pred)

if CRF:
    l=[]
    for image, mask in zip(tqdm(x_valid, desc='CRF'), y_pred):
        crf_output = crf(image, mask)
        l.append(crf_output)

    y_pred = np.array(l)
    print('After CRF, iou:', iou_metric_batch(y_valid, y_pred))

    print('plot prediction')
    plot_prediction(x_valid, y_valid, y_pred, fname='sanity_check_prediction_CRF.png')


