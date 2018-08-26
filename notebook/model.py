import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.random.seed(1)

import tensorflow as tf
tf.set_random_seed(2)

from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, PReLU, ReLU, UpSampling2D, Concatenate, Add
from keras.layers.core import Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras import regularizers
from metrics import iou_metric, iou_metric_batch

# Losses
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def mixed_dice_bce_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + 5*dice_loss(y_true, y_pred)

def iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value

def focal_loss(y_true, y_pred):
    gamma=1.0
    alpha=0.25
    epsilon = 1e-8
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    pred_pt = tf.where(tf.equal(y_true_f, 1), y_pred_f, 1-y_pred_f)
    alpha_t = tf.scalar_mul(alpha, tf.ones_like(y_true_f, dtype=tf.float32))
    alpha_t = tf.where(tf.equal(y_true_f, 1), alpha_t, 1-alpha_t)
    losses = K.mean(-alpha_t * K.pow(1. - pred_pt, gamma) * K.log(pred_pt+epsilon))
    return losses

# Model
def VGGUnet(img_shape, start_filter=32, out_ch=1, bn=True, up=False, dro=0.5):
    def _conv_block(input_, filter, nb_conv, bn=False, dro=0.5):
        x = input_
        for _ in range(nb_conv):
            x = Conv2D(filter, (3, 2), activation='relu', padding='same')(x)
            x = BatchNormalization()(x) if bn else x
            n = Dropout(dro)(x) if dro else x
        return x


    def _up_block(input1, input2, filter, upconv=False):
        x = input1
        if upconv:
            x = UpSampling2D()(x)
            x = Conv2D(filter, (3, 2), activation='relu', padding='same')(x)
        else:
            x = Conv2DTranspose(filter, 3, strides=2, activation='relu', padding='same')(x)
        x = Concatenate()([x, input2])
        x = _conv_block(x, filter, nb_conv=1, dro=0.5)
        return x

    i = Input(shape=img_shape)
    s = BatchNormalization() (i)
    # Block 1
    x = _conv_block(s, filter=start_filter, nb_conv=2, bn=bn, dro=dro)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f1 = x
        # Block 2
    x = _conv_block(x, filter=start_filter*2, nb_conv=2, bn=bn, dro=dro)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f2 = x
        # Block 3
    x = _conv_block(x, filter=start_filter*4, nb_conv=3, bn=bn, dro=dro)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f3 = x
        # Block 4
    x = _conv_block(x, filter=start_filter*8, nb_conv=3, bn=bn, dro=dro)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    f4 = x

    o = _conv_block(f4, filter=start_filter*8, nb_conv=1, bn=bn, dro=0)
    o = _up_block(o, f3, start_filter*4)
    o = _up_block(o, f2, start_filter*2)
    o = _up_block(o, f1, start_filter)
    o = _up_block(o, i, start_filter//2)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)


def conv_block(m, dim, acti, bn, res, do=0):
	n = Conv2D(dim, 3, activation=acti, padding='same')(m)
	n = BatchNormalization()(n) if bn else n
	n = Dropout(do)(n) if do else n
	n = Conv2D(dim, 3, activation=acti, padding='same')(n)
	n = BatchNormalization()(n) if bn else n
	return Concatenate()([m, n]) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = Concatenate()([n, m])
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m

def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.5, batchnorm=False, maxpool=True, upconv=True, residual=False):
    i = Input(shape=img_shape)
    o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
    o = Conv2D(out_ch, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)


def create_model(img_shape, vggunet, start_ch, depth=0, maxpool=True, upconv=True, residual=False):
    K.clear_session()
    if vggunet:
        model = VGGUnet(img_shape=img_shape, start_filter=start_ch, up=upconv)
    else:
        model = UNet(img_shape=img_shape, start_ch=start_ch, depth=depth, batchnorm=True, residual=residual, maxpool=maxpool, upconv=upconv)
    model.compile(optimizer='adam', loss=mixed_dice_bce_loss, metrics=[iou_metric])
    return model


def fit(model, X_train, Y_train, x_valid, y_valid, output_name):
    early_stopping = EarlyStopping(patience=20, verbose=1, monitor='val_iou_metric', mode='max')

    checkpointer = ModelCheckpoint(output_name, save_best_only=True, verbose=1, monitor='val_iou_metric', mode='max')
    checkpointer2 = ModelCheckpoint('loss_'+output_name, save_best_only=True, verbose=1)

    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    csvlog = CSVLogger('{}_log.csv'.format(output_name.split('.')[0]))

    results = model.fit(X_train, Y_train, validation_data=[x_valid, y_valid], batch_size=32, epochs=200,
                        callbacks=[checkpointer,checkpointer2, reduce_lr, csvlog, early_stopping], verbose=1, shuffle=True)
    return results

"""
to test : https://github.com/nicolov/segmentation_keras/blob/master/model.py
to test : https://github.com/ternaus/TernausNet
to test : https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py

"""
