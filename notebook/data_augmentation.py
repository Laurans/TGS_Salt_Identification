from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from imgaug import augmenters as iaa
import imgaug as ia
from skimage import img_as_float, exposure
from scipy import misc, ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import reconstruction, disk
from skimage import exposure

from skimage.morphology import dilation, disk, square
from keras.applications.vgg16 import preprocess_input

import cv2
import numpy as np

from tqdm import tqdm, trange
import sys

def apply_features(image):
    lpb_large = exposure.equalize_hist(image)
    return np.dstack((image, lpb_large))

def augment_images(x_train, y_train):
    all_x = []
    all_y = []

    aug_list = [iaa.Noop(), iaa.Fliplr(1)]#, iaa.Flipud(1)]
    #aug_list.append(iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)]))
    
    for augmentor in tqdm(aug_list):
        aug_imgs = []
        labels_imgs = []
        for i in trange(x_train.shape[0]):
            deterministic = augmentor.to_deterministic()
            aug_img = deterministic.augment_image(x_train[i])
            label_img = deterministic.augment_image(y_train[i])

            aug_imgs.append(aug_img)
            labels_imgs.append(label_img)

        all_x.append(np.array(aug_imgs))
        all_y.append(np.array(labels_imgs))

    x_train_ = np.vstack(all_x)
    y_train_ = np.vstack(all_y)

    print()
    print('Augment images done')
    return x_train_, y_train_

def plot_list(images=[], labels=[]):
    n_img = len(images)
    n_lab = len(labels)
    n = n_lab + n_img
    fig, axs = plt.subplots(1, n, figsize=(16, 12))
    for i, image in enumerate(images):
        axs[i].imshow(image)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    for j, label in enumerate(labels):
        axs[n_img + j].imshow(label, cmap='nipy_spectral')
        axs[n_img + j].set_xticks([])
        axs[n_img + j].set_yticks([])
    plt.show()
