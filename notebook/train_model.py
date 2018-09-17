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

DATA = True
MODEL = True
LOAD_PREV_MODEL = False
TRAIN = True
PRED = False
SANITY_CHECK_IOU = False
CRF = False
HIST = True
PRED_ON_TRAIN = False


datamanager = DataManager()

if DATA:
    print('Loading dataset')

    start_time = datetime.datetime.now()
    #(x_train, y_train), (x_valid, y_valid) = datamanager.load_dataset()
    X_train, Y_train, coverage, _ = datamanager.load_train()

    x_train, x_valid, y_train, y_valid, ids_train, ids_valid, cov_train, cov_valid = train_test_split(
        X_train, Y_train, np.array(datamanager.train_ids), coverage, test_size=0.15, stratify=coverage[:, 1], random_state=12)

    x_train_, y_train_ = augment_images(x_train, y_train)

    if PRED_ON_TRAIN:
        x_valid = x_train_
        y_valid = y_train_
        ids_valid = ids_train

    y_train_ = np.piecewise(y_train_, [y_train_ > 127.5, y_train_ < 127.5], [1, 0])
    y_valid = np.piecewise(y_valid, [y_valid > 127.5, y_valid < 127.5], [1, 0])

    assert x_train_.shape[1:] == x_valid.shape[1:]
    time_delta = datetime.datetime.now() - start_time
    print('Loading time', time_delta)
    
if MODEL:
    amodel = create_model((datamanager.img_size_input, datamanager.img_size_input, datamanager.im_chan), start_ch=32, depth=5, repetitions=[1, 1, 1, 1, 1],
    filter_sizes=[3, 3, 3, 3, 3])
    amodel.summary()
    if LOAD_PREV_MODEL:
        amodel.load_weights('model.h5')
    if TRAIN:
        history = fit(amodel, x_train_, y_train_, x_valid, y_valid, 'model.h5')

if PRED:
    model = load_model('model.h5', custom_objects={'mixed_dice_bce_loss': mixed_dice_bce_loss, 'dice_loss': dice_loss, 'iou_metric':iou_metric, 'focal_loss':focal_loss, 'Scale': Scale})
    
    if SANITY_CHECK_IOU:
        preds_valid = model.predict(x_valid, verbose=1)
        iou_best, threshold_best = best_iou_and_threshold(y_valid, preds_valid, shortcut=True)
        print('Sanity Check, iou', iou_best)

    else:
        tta_model = TTA_ModelWrapper(model)
        preds_valid = tta_model.predict(x_valid)
        iou_best, threshold_best = best_iou_and_threshold(y_valid, preds_valid, shortcut=True)
        print('TTA', iou_best)
        y_pred = np.int32(preds_valid > 0.5)

        if CRF:
            l=[]
            for image, mask in zip(tqdm(x_valid, desc='CRF'), y_pred):
                crf_output = crf(image, mask)
                l.append(crf_output)

            y_pred_spe = np.array(l)
            print('After CRF, iou:', iou_metric_batch(y_valid, y_pred_spe))

            #print('plot prediction')
            #plot_prediction(x_valid_down, y_valid_down, y_pred_down, fname='sanity_check_prediction_CRF.png')


        print('plot_prediction')
        x_valid_down = x_valid[:,:,:, 0]
        y_valid_down = y_valid
        y_pred_down = y_pred
        plot_prediction(x_valid_down, y_valid_down, y_pred_down)

        if HIST:
            iou_scores = plot_hist(y_valid_down, y_pred_down)
            indexes = np.array(iou_scores) < 0.4
            plot_prediction(x_valid_down[indexes], y_valid_down[indexes], y_pred_down[indexes], fname='hard_images.png', image_id=ids_valid[indexes])  

