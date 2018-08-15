from data_io import *
from data_augmentation import *
import joblib

from sklearn.model_selection import train_test_split

datamanager = DataManager()

X_train, Y_train, coverage = datamanager.load_train()

x_train, x_valid, y_train, y_valid, cov_train, cov_valid = train_test_split(
    X_train, Y_train, coverage, test_size=0.2, stratify=coverage[:, 1], random_state=24)

x_train_, y_train_ = augment_images(x_train, y_train)

y_train_ = np.piecewise(y_train_, [y_train_ > 127.5, y_train_ < 127.5], [1, 0])

y_valid = np.piecewise(y_valid, [y_valid > 127.5, y_valid < 127.5], [1, 0])
# Checkpoint
datamanager.save_dataset([(x_train_, y_train_), (x_valid, y_valid)])
