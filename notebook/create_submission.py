from data_io import *
from model import *
import pandas as pd
from tqdm import tqdm
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
        p0 = self.model.predict(X, verbose=1)

        x1 = np.array([np.fliplr(i) for i in X])
        p1 = self.model.predict(x1, verbose=1)
        p1 = np.array([np.fliplr(i) for i in p1])

        x2 = np.array([np.flipud(i) for i in X])
        p2 = self.model.predict(x2, verbose=1)
        p2 = np.array([np.flipud(i) for i in p2])

        x3 = np.array([np.fliplr(i) for i in x2])
        p3 = self.model.predict(x3, verbose=1)
        p3 = np.array([np.fliplr(np.flipud(i)) for i in p3])

        p = (p0 + p1 + p2 + p3 ) / 4
        return p

    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)


datamanager = DataManager()

X_test = datamanager.load_test()

p = []
for m in tqdm([2, 5, 6, 7, 8], desc='pred by model'):
    model = load_model('model_{}.h5'.format(m), custom_objects={'mixed_dice_bce_loss': mixed_dice_bce_loss, 'dice_loss': dice_loss})
    tta_model = TTA_ModelWrapper(model)
    pred = tta_model.predict(X_test)
    p.append(pred)

print('Now mean')
pred = np.zeros_like(pred)
for prediction in p:
    pred += prediction

pred = pred / len(p)
print('Now threshold')

thres =  0.5
preds_test = (pred > thres).astype(np.uint8)
"""
l = []
for image in tqdm(X_test):
    mask = (tta_model.predict(image) > thres).astype(np.uint8)
    crf_output = crf(image, mask)
    l.append(mask)

preds_test = np.array(l)
"""
print('pred_test shape', preds_test.shape)

pred_downsampled = datamanager.downsample(preds_test)

pred_dict = {fn[:-4]:RLenc(np.round(pred_downsampled[i])) for i,fn in enumerate(tqdm(datamanager.test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
