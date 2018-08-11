from data_io import *
from data_augmentation import *
from metrics import *
from model import *
import pandas as pd

datamanager = DataManager()

model = load_model('model_testing_script_0.h5')

X_test = datamanager.load_test()

print(X_test.shape)

preds_test = (model.predict(X_test, verbose=1) > 0.59).astype(np.uint8)

pred_downsampled = datamanager.downsample(preds_test)

pred_dict = {fn[:-4]:RLenc(np.round(pred_downsampled[i])) for i,fn in enumerate(tqdm(datamanager.test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
