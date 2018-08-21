from data_io import *
from model import *
import pandas as pd

import numpy as np

class TTA_ModelWrapper():
    """A simple TTA wrapper for keras computer vision models.
    Args:
        model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Wraps the predict method of the provided model.
        Augments the testdata with horizontal and vertical flips and
        averages the results.
        Args:
            X (numpy array of dim 4): The data to get predictions for.
        """
        X = np.expand_dims(X, 0)
        p0 = self.model.predict(X)
        p1 = self.model.predict(np.fliplr(X))
        p2 = self.model.predict(np.flipud(X))
        p3 = self.model.predict(np.fliplr(np.flipud(X)))
        p = (p0 +
             np.fliplr(p1) +
             np.flipud(p2) +
            np.fliplr(np.flipud(p3))
             ) / 4
        return p[0]

    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)


datamanager = DataManager()

model = load_model('model.h5', custom_objects={'mixed_dice_bce_loss': mixed_dice_bce_loss, 'multiclass_dice_loss': multiclass_dice_loss})
tta_model = TTA_ModelWrapper(model)

X_test = datamanager.load_test()

thres =  0.5
l = []
for image in tqdm(X_test):
    mask = (tta_model.predict(image) > thres).astype(np.uint8)
    crf_output = crf(image, mask)
    l.append(crf_output)

preds_test = np.array(l)
print('pred_test shape', preds_test.shape)

pred_downsampled = datamanager.downsample(preds_test)

pred_dict = {fn[:-4]:RLenc(np.round(pred_downsampled[i])) for i,fn in enumerate(tqdm(datamanager.test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
