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
(x_train, y_train), (x_valid, y_valid) = datamanager.load_dataset()
time_delta = datetime.datetime.now() - start_time
print('Loading time', time_delta)

amodel = create_model(datamanager.im_height, datamanager.im_width, datamanager.im_chan)
amodel.load_weights('model.h5')

history = fit(amodel, x_train, y_train, x_valid, y_valid, 'model.h5')

model = load_model('model.h5', custom_objects={'mean_iou': mean_iou, 'mixed_dice_bce_loss': mixed_dice_bce_loss, 'multiclass_dice_loss': multiclass_dice_loss})
preds_valid = model.predict(x_valid, verbose=1)

thresholds = np.linspace(0, 1, 50)
ious = np.array([iou_metric_batch(y_valid, np.int32(preds_valid > threshold)) for threshold in tqdm(thresholds, desc='compute ious for thres')])

threshold_best_index = np.argmax(ious[9:-10]) + 9
iou_best = ious[threshold_best_index]
threshold_best = thresholds[threshold_best_index]

print("Threshold ", threshold_best, 'Ious', iou_best)
