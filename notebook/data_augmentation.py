from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from imgaug import augmenters as iaa
from skimage import img_as_float, exposure
from scipy import misc, ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import reconstruction, disk
from skimage.filters import rank

from skimage.morphology import label

import cv2
import numpy as np

from tqdm import tqdm

def filtering_regional_maxima(img):
    image = img_as_float(img)
    image = ndimage.gaussian_filter(image, 1)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method='dilation') * 255
    return np.array(dilated, dtype=np.uint8)

def global_equalize(img):
    return np.array(exposure.equalize_hist(img) * 255, dtype=np.uint8)

def elastic_transform(image, alpha, sigma, seed=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    if seed is None:
        random_state = np.random.RandomState()
    else:
        random_state = np.random.RandomState(seed=seed)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


def augment_images(x_train, y_train):
    x_train = np.append(x_train, np.array( [np.fliplr(x) for x in x_train]), 0)
    y_train = np.append(y_train, np.array( [np.fliplr(x) for x in y_train]), 0)

    #x_train_ = np.append(x_train, np.array( [filtering_regional_maxima(x) for x in tqdm(x_train)]), 0)
    #y_train_ = np.vstack([y_train, y_train.copy()])

    #x_train_ = np.append(x_train_, np.array([global_equalize(x) for x in tqdm(x_train)]), 0)
    #y_train_ = np.vstack([y_train_, y_train.copy()])

    #x_train_ = np.append(x_train_, np.array([np.expand_dims(elastic_transform(x.squeeze(), 20, 4, 20), -1)
    #                              for x in tqdm(x_train)]), 0)
    #y_train_ = np.append(y_train_, np.array([np.expand_dims(elastic_transform(x.squeeze(), 20, 4, 20), -1)
    #                              for x in tqdm(y_train)]), 0)
    x_train_ , y_train_ = x_train, y_train
    print('Augment images done')
    return x_train_, y_train_
