from data_io import *
from model import *
from metrics import *
from losses import *
from data_augmentation import *

import pandas as pd
from tqdm import tqdm
import numpy as np
import keras.backend as K
import joblib
import os

custom_objects={
        'mixed_dice_bce_loss': mixed_dice_bce_loss, 
        'dice_loss': dice_loss, 
        'iou_metric':iou_metric, 
        'focal_loss':focal_loss,
        'mixed_dice_bce_loss_masked': mixed_dice_bce_loss_masked,
        'lovasz_loss': lovasz_loss,
        'mixed_bce_lovasz': mixed_bce_lovasz,
        }
datamanager = DataManager()

def prepare_prediction_from_unet(fold, interval, flat):
    list_models = ['fold_{}_model_{}.h5'.format(fold, i) for i in interval]

    X_test = datamanager.load_test()
    for name in list_models:
        model = load_model(name, custom_objects=custom_objects)
        tta_model = TTA_ModelWrapper(model)
        pred = tta_model.predict(X_test)
        if flat:
            pred = pred.reshape(pred.shape[0], -1)

        for a in range(0, pred.shape[0], 500):
            start = a
            end = a+500
            if not os.path.exists('../data/prediction/fold_{}'.format(fold)): # dont exist create
                os.mkdir('../data/prediction/fold_{}'.format(fold))
            
            if not os.path.exists('../data/prediction/fold_{}/part_{}'.format(fold, start)):
                os.mkdir('../data/prediction/fold_{}/part_{}'.format(fold, start))

            joblib.dump(pred[start:end], '../data/prediction/fold_{}/part_{}/{}.pkl'.format(fold, start, name))

def generator_unet(fold, max_models):
    nb_parts = len(os.listdir('../data/prediction/fold_{}/'.format(fold)))
    part = 0
    while part < nb_parts * 500:
        path = '../data/prediction/fold_{}/part_{}/'.format(fold, part)
        file_list = os.listdir(path)

        assert len(file_list) == max_models
        x = []
        for name in tqdm(file_list, disable=True):
            pred = joblib.load(path+name)
            x.append(pred)
        x = np.dstack(x)
        x = x.reshape(x.shape[0], -1, 1)
        yield x
        part += 500 

def get_stacking_pred(fold, max_models):
    nb_parts = len(os.listdir('../data/prediction/fold_{}/'.format(fold)))

    model = load_model('fold_{}_stacking.h5'.format(fold), custom_objects=custom_objects)
    pred = model.predict_generator(generator_unet(fold, max_models), steps=nb_parts, verbose=1)
    return pred

#prepare_prediction_from_unet(1, range(1, 7), True)
pred = get_stacking_pred(1, 6)
thres =  0.50
preds_test = (pred > thres).astype(np.uint8)
print('pred_test shape', preds_test.shape)

pred_downsampled = preds_test

pred_dict = {idx[:-4]: rle_encode(np.round(pred_downsampled[i])) for i, idx in enumerate(tqdm(datamanager.test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')