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
TRAIN = False
TRAIN_STACKING = True
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
        X_train, Y_train, np.array(datamanager.train_ids), coverage, test_size=0.08, stratify=coverage[:, 1], random_state=12)

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
    amodel = create_model((datamanager.img_size_input, datamanager.img_size_input, datamanager.im_chan))
    amodel.summary()
    if LOAD_PREV_MODEL:
        print('Loading weights')
        amodel.load_weights('model.h5')
    if TRAIN:
        print('Start training')
        for i, finetune, loss in zip(range(1, 5), [False, True, True, True], ['mixed3', 'mixed2', 'mixed', 'lovasz']):
            fit(amodel, x_train_, y_train_, x_valid, y_valid, 'model_{}.h5'.format(i), finetune, loss)

    if TRAIN_STACKING:
        a = []
        b = []
        for i in range(1, 5):
            model = load_model('model_{}.h5'.format(i), 
            custom_objects={
                'mixed_dice_bce_loss': mixed_dice_bce_loss, 
                'dice_loss': dice_loss, 
                'iou_metric':iou_metric, 
                'focal_loss':focal_loss,
                'mixed_dice_bce_loss_masked': mixed_dice_bce_loss_masked,
                'lovasz_loss': lovasz_loss,
                'mixed_bce_lovasz': mixed_bce_lovasz,
                })
            tta_model = TTA_ModelWrapper(model)
            pred = tta_model.predict(x_train_)
            pred = pred.reshape(pred.shape[0], -1)
            a.append(pred)
            pred = tta_model.predict(x_valid)
            pred = pred.reshape(pred.shape[0], -1)
            b.append(pred)

        xt = np.dstack(a)
        xt = xt.reshape(xt.shape[0], -1, 1)
        xv = np.dstack(b)
        xv = xv.reshape(xv.shape[0], -1, 1)

        stack_model = stacking(xt.shape[1:], len(a))
        fit(stack_model, xt, y_train_, xv, y_valid, 'stacking.h5', False, 'mixed')

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

