from data_io import *
from data_augmentation import *
from metrics import *
from model import *

from sklearn.model_selection import train_test_split

datamanager = DataManager()

X_train, Y_train, coverage = datamanager.load_train()

params = [(16, 5, False, 24), (16, 5, True, 42),
          (16, 6, False, 78), (16, 6, True, 87),
          (32, 4, False, 69), (32, 4, True, 96),
          (32, 5, False, 12), (32, 5, True, 21),
          (64, 4, False, 17), (64, 4, True, 71)]

for i, (start, depth, res, seed) in enumerate(params):
    print('-'*10)
    print('Start {} model'.format(i))
    print('-'*10)
    x_train, x_valid, y_train, y_valid, cov_train, cov_valid = train_test_split(
        X_train, Y_train, coverage, test_size=0.15, stratify=coverage[:, 1], random_state=seed)

    x_train_, y_train_ = augment_images(x_train, y_train)

    y_train_ = np.piecewise(y_train_, [y_train_ > 127.5, y_train_ < 127.5], [1, 0])

    y_valid = np.piecewise(y_valid, [y_valid > 127.5, y_valid < 127.5], [1, 0])

    amodel = create_model(datamanager.im_height, datamanager.im_width, datamanager.im_chan, start=start, depth=depth, residual=res)
    history = fit(amodel, x_train_, y_train_, x_valid, y_valid, 'model_{}.h5'.format(i))


    print('-'*10)
    print('End {} model'.format(i))
    print('-'*10)
