import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, PReLU, ReLU
from keras.layers.core import Lambda, RepeatVector
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras import regularizers
import numpy as np


def multiclass_dice_loss(output, target):
    return 1 - (2* K.sum(output*target)) / (K.sum(output) + K.sum(target) + 1e-7)

def mixed_dice_bce_loss(y_true, y_pred, dice_weight=5.0, bce_weight=1.0):
    bce_loss = binary_crossentropy(y_true=y_true, y_pred=y_pred)
    dice_loss = multiclass_dice_loss(y_pred, y_true)
    return dice_weight*dice_loss+bce_weight*bce_loss


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def create_model(im_height, im_width, im_chan):
    inputs = Input((im_height, im_width, im_chan))
    s = Lambda(lambda x: x/255)(inputs)

    x = Conv2D(16, (5, 5), activation=None, padding='same', name='block1_conv1', kernel_regularizer=regularizers.l2(0.0001)) (s)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    x = Conv2D(16, (5, 5), activation=None, padding='same', name='block1_conv2', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = Dropout(0.1)(x)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    c1 = concatenate([s, x])
    p1 = MaxPooling2D((2, 2), name='block1_pool') (c1)

    x = Conv2D(32, (5, 5), activation=None, padding='same', name='block2_conv1', kernel_regularizer=regularizers.l2(0.0001)) (p1)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(32, (5, 5), activation=None, padding='same', name='block2_conv2', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    c2 = concatenate([p1, x])
    p2 = MaxPooling2D((2, 2)) (c2)

    x = Conv2D(64, (5, 5), activation=None, padding='same', name='block3_conv1', kernel_regularizer=regularizers.l2(0.0001)) (p2)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (5, 5), activation=None, padding='same', name='block3_conv2', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (5, 5), activation=None, padding='same', name='block3_conv3', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    c3 = concatenate([p2, x])
    p3 = MaxPooling2D((2, 2)) (c3)

    x = Conv2D(128, (5, 5), activation=None, padding='same',  name='block4_conv1', kernel_regularizer=regularizers.l2(0.0001)) (p3)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (5, 5), activation=None, padding='same',  name='block4_conv2', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (5, 5), activation=None, padding='same',  name='block4_conv3', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    c4 = concatenate([p3, x])
    p4 = MaxPooling2D((2, 2)) (c4)

    x = Conv2D(256, (5, 5), activation=None, padding='same', name='block5_conv1', kernel_regularizer=regularizers.l2(0.0001)) (p4)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(256, (5, 5), activation=None, padding='same', name='block5_conv2', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(256, (5, 5), activation=None, padding='same', name='block5_conv3', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    c5 = concatenate([p4, x])
    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c5)

    x = concatenate([u6, c4])
    x = Conv2D(256, (1, 1), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)

    x = Conv2D(256, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(256, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    c6 = concatenate([u6, x])
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (c6)

    x = concatenate([u7, c3])
    x = Conv2D(128, (1, 1), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)

    x = Conv2D(128, (5, 5), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (5, 5), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(128, (5, 5), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    c7 = concatenate([u7, x])
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c7)

    x = concatenate([u8, c2])
    x = Conv2D(64, (1, 1), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)

    x = Conv2D(64, (5, 5), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(64, (5, 5), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    c8 = concatenate([u8, x])
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c8)


    x = concatenate([u9, c1], axis=3)
    x = Conv2D(32, (1, 1), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)

    x = Conv2D(32, (5, 5), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    x = PReLU()(x)
    x = Dropout(0.1)(x)
    x = Conv2D(32, (5, 5), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    x = concatenate([u9, x])
    x = Conv2D(16, (1, 1), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
    x = BatchNormalization() (x)
    c9 = PReLU()(x)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
    outputs = Dropout(0.1)(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])
    optimizer = Adam(lr=0.0001, amsgrad=True)
    model.compile(optimizer=optimizer, loss=mixed_dice_bce_loss, metrics=[mean_iou])
    return model


def fit(model, X_train, Y_train, x_valid, y_valid, output_name):
    checkpointer = ModelCheckpoint(output_name, monitor='val_mean_iou', mode='max', save_best_only=True, period=4, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)
    csvlog = CSVLogger('{}_log.csv'.format(output_name.split('.')[0]))

    results = model.fit(X_train, Y_train, validation_data=[x_valid, y_valid], batch_size=32, epochs=100,
                        callbacks=[checkpointer, reduce_lr, csvlog], verbose=1)
    return results

"""
to test : https://github.com/nicolov/segmentation_keras/blob/master/model.py
to test : https://github.com/ternaus/TernausNet
to test : https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py

"""
