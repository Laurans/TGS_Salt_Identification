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

affine_seq = iaa.Sequential([
    # General
    iaa.SomeOf((1, 2),
               [iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-10, 10),
                           translate_percent={"x": (-0.25, 0.25)}, mode='symmetric'),
                ]),
    # Deformations
    iaa.Sometimes(0.3, iaa.PiecewiseAffine(scale=(0.04, 0.08))),
    iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.05, 0.1))),
], random_order=True)

intensity_seq = iaa.Sequential([
    iaa.Invert(0.3),
    iaa.Sometimes(0.3, iaa.ContrastNormalization((0.5, 1.5))),
    iaa.OneOf([
        iaa.Noop(),
        iaa.Sequential([
            iaa.OneOf([
                iaa.Add((-10, 10)),
                iaa.AddElementwise((-10, 10)),
                iaa.Multiply((0.95, 1.05)),
                iaa.MultiplyElementwise((0.95, 1.05)),
            ]),
        ]),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.AverageBlur(k=(2, 5)),
            iaa.MedianBlur(k=(3, 5))
        ])
    ])
], random_order=False)

crop_pad_seq = iaa.Sequential([
    affine_seq,
    iaa.Sometimes(0.5, iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
    iaa.Sometimes(0.3, iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ))
])

def augment_images(x_train, y_train):

    AUG_NR = 6

    all_x = []
    all_y = []

    sys.stdout.flush()
    for _ in trange(AUG_NR, desc='AUG NR'):
        aug_imgs = []
        labels_imgs = []
        for i in trange(x_train.shape[0], desc='aug each images'):
            augmentor = affine_seq.to_deterministic()
            aug_img = augmentor.augment_image(x_train[i])
            label_img = augmentor.augment_image(y_train[i])
            aug_imgs.append(aug_img)
            labels_imgs.append(label_img)

        all_x.append(np.array(aug_imgs))
        all_y.append(np.array(labels_imgs))

    for _ in trange(AUG_NR, desc='AUG NR'):
        aug_imgs = []
        labels_imgs = []
        for i in trange(x_train.shape[0], desc='aug each images'):
            augmentor = affine_seq.to_deterministic()
            crop_pad_seq
            aug_img = intensity_seq.augment_image(augmentor.augment_image(x_train[i]))
            label_img = augmentor.augment_image(y_train[i])
            aug_imgs.append(aug_img)
            labels_imgs.append(label_img)

        all_x.append(np.array(aug_imgs))
        all_y.append(np.array(labels_imgs))

    for _ in trange(AUG_NR, desc='AUG NR'):
        aug_imgs = []
        labels_imgs = []
        for i in trange(x_train.shape[0], desc='aug each images'):
            augmentor = crop_pad_seq.to_deterministic()
            aug_img = augmentor.augment_image(x_train[i])
            label_img = augmentor.augment_image(y_train[i])
            aug_imgs.append(aug_img)
            labels_imgs.append(label_img)

        all_x.append(np.array(aug_imgs))
        all_y.append(np.array(labels_imgs))

    x_train_ = np.vstack(all_x)
    y_train_ = np.vstack(all_y)

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
