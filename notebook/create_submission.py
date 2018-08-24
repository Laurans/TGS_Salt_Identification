from data_io import *
from model import *
import pandas as pd
from tqdm import tqdm
import numpy as np
import keras.backend as K

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
        p = self.model.predict(X, verbose=1)

        x1 = np.array([np.fliplr(i) for i in X])
        p1 = self.model.predict(x1, verbose=1)
        p += np.array([np.fliplr(i) for i in p1])

        x1 = np.array([np.flipud(i) for i in X])
        p1 = self.model.predict(x1, verbose=1)
        p += np.array([np.flipud(i) for i in p1])

        x1 = np.array([np.fliplr(i) for i in x1])
        p1 = self.model.predict(x1, verbose=1)
        p += np.array([np.fliplr(np.flipud(i)) for i in p1])

        p /= 4
        return p

    def _expand(self, x):
        return np.expand_dims(np.expand_dims(x, axis=0), axis=3)


datamanager = DataManager()

X_test = datamanager.load_test()

pred = np.zeros(X_test.shape)
n = 0

bootstrap1 = [2, 5, 6, 7, 8]
bootstrap2 = [6, 2, 0, 4, 7]

for e, bootstrap in enumerate([bootstrap1, bootstrap2]):
    for m in tqdm(bootstrap, desc='pred by model'):
        model = load_model('model_archive/bootstrap_{}/model_{}.h5'.format(e+1, m), custom_objects={'mixed_dice_bce_loss': mixed_dice_bce_loss, 'dice_loss': dice_loss})
        tta_model = TTA_ModelWrapper(model)
        pred += tta_model.predict(X_test)
        n += 1
        K.clear_session()
print('Now mean')
pred /= n
print('Now threshold')

thres =  0.5
preds_test = (pred > thres).astype(np.uint8)
"""
l = []
for image, mask in zip(tqdm(X_test), preds_test):
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
