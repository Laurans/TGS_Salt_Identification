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

def get_classes(x):
    all_cov = []
    for i in trange(x.shape[0]):
        cov_img = np.sum(x[i]/255)
        if cov_img == 1:
            all_cov.append([0, 0, 1])
        elif cov_img == 0:
            all_cov.append([1, 0, 0])
        else:
            all_cov.append([0, 1, 0])
    
    return np.array(all_cov)

def augment_images(x_train, y_train):
    all_x = []
    all_y = []

    aug_list = [iaa.Noop(), iaa.Fliplr(1)] # Good
    aug_list.append(iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, mode='reflect')) # Good
    aug_list.append(iaa.Sequential([iaa.Fliplr(1), iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, mode='reflect')])) # Good
    
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
