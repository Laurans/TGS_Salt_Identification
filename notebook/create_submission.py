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

X_test, _ = datamanager.load_test()

x = []

list_models = ['model_1.h5','model_2.h5', 'model_3.h5', 'model_4.h5']
for name in list_models:
    model = load_model(name, custom_objects=custom_objects)
    tta_model = TTA_ModelWrapper(model)
    pred = tta_model.predict(X_test)
    pred = pred.reshape(pred.shape[0], -1)
    x.append(pred)

x = np.dstack(x)
x = x.reshape(x.shape[0], -1, 1)
model = load_model('stacking.h5', custom_objects=custom_objects)
pred = model.predict(x)

#pred = p/len(list_models)
thres =  0.5
preds_test = (pred > thres).astype(np.uint8)
print('pred_test shape', preds_test.shape)

pred_downsampled = preds_test

pred_dict = {idx[:-4]: rle_encode(np.round(pred_downsampled[i])) for i, idx in enumerate(tqdm(datamanager.test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
