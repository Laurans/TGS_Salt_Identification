from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from imgaug import augmenters as iaa
import imgaug as ia
from skimage import img_as_float, exposure
from scipy import misc, ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import reconstruction, disk
from skimage.filters import rank

from skimage.morphology import label
from keras.applications.vgg16 import preprocess_input

import cv2
import numpy as np

from tqdm import tqdm, trange
import sys

double_flip = iaa.Sequential([iaa.Fliplr(1.0), iaa.Flipud(1.0)])

def augment_images(x_train, y_train, with_constrast=False, with_noise=False):
    all_x = [x_train]
    all_y = [y_train]

    aug_list = [iaa.Fliplr(1), iaa.Flipud(1), double_flip]

    if with_constrast:
        aug_list.append(iaa.ContrastNormalization(alpha=1.5))

    if with_noise:
        aug_list.append(iaa.AdditiveGaussianNoise(scale=0.05*255))
    for augmentor in tqdm(aug_list):
        aug_imgs = []
        labels_imgs = []
        for i in trange(x_train.shape[0]):
            aug_img = augmentor.augment_image(x_train[i])
            label_img = augmentor.augment_image(y_train[i])
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
