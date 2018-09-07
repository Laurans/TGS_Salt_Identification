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

datamanager = DataManager()

X_test, _ = datamanager.load_test()

model = load_model('model.h5', custom_objects={
    'mixed_dice_bce_loss': mixed_dice_bce_loss, 
    'dice_loss': dice_loss, 
    'iou_metric':iou_metric, 
    'focal_loss':focal_loss,
    'Scale': Scale})
tta_model = TTA_ModelWrapper(model)
pred = tta_model.predict(X_test)

thres =  0.5
preds_test = (pred > thres).astype(np.uint8)
print('pred_test shape', preds_test.shape)

pred_downsampled = datamanager.downsample(preds_test)

pred_dict = {idx[:-4]: rle_encode(np.round(pred_downsampled[i])) for i, idx in enumerate(tqdm(datamanager.test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
