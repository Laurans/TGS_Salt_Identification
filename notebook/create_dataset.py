from data_io import *
from data_augmentation import *
import joblib

from sklearn.model_selection import train_test_split


def create_dataset_aug():
    datamanager = DataManager()

    X_train, Y_train, coverage = datamanager.load_train()
    x_train, x_valid, y_train, y_valid, _, _ = train_test_split(
        X_train, Y_train, coverage, test_size=0.15, stratify=coverage[:, 1], random_state=24)

    x_train_, y_train_ = augment_images(x_train, y_train)

    y_train_ = np.piecewise(y_train_, [y_train_ > 127.5, y_train_ < 127.5], [1, 0])

    y_valid = np.piecewise(y_valid, [y_valid > 127.5, y_valid < 127.5], [1, 0])
    # Checkpoint
    datamanager.save_dataset([(x_train_, y_train_), (x_valid, y_valid)])

def create_dataset_on_mask():
    datamanager = DataManager()

    X_train, Y_train, coverage = datamanager.load_train()
    counter = 0

    new_y = np.zeros((len(Y_train),), dtype=np.uint8)
    for i, y in enumerate(Y_train):
        if y.sum() == 0:
            counter += 1
        else:
            new_y[i] = 1

    x_train, x_valid, y_train, y_valid = train_test_split(
        X_train, new_y, test_size=0.15, stratify=new_y, random_state=24)

    datamanager.save_dataset([(x_train, y_train), (x_valid, y_valid)], name='binary')
    print('Il y a {} masques vides soit {:.3f}\% du dataset'.format(counter, 100*counter/len(Y_train)))
