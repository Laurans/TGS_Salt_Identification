from data_io import *
from model import *
import pandas as pd
from model import mean_iou

datamanager = DataManager()

model = load_model('model_big.h5', custom_objects={'mean_iou': mean_iou, 'mixed_dice_bce_loss': mixed_dice_bce_loss, 'multiclass_dice_loss': multiclass_dice_loss})

X_test = datamanager.load_test()

print(X_test.shape)
thres =  0.244
preds_test = (model.predict(X_test, verbose=1) > thres).astype(np.uint8)

pred_downsampled = datamanager.downsample(preds_test)

pred_dict = {fn[:-4]:RLenc(np.round(pred_downsampled[i])) for i,fn in enumerate(tqdm(datamanager.test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
