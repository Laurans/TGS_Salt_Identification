import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.random.seed(1)

import tensorflow as tf
tf.set_random_seed(2)

from keras.models import Model, load_model
from keras.layers import *
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
from metrics import best_iou_and_threshold
from keras.regularizers import l2
from losses import *

# Callback
class ValGlobalMetrics(Callback):
    def __init__(self, x_valid, y_valid):
        self.x_valid = x_valid
        self.y_valid = y_valid

    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.x_valid))
        best_iou, _ = best_iou_and_threshold(y_true=self.y_valid, y_pred=predict, shortcut=True)
        logs['val_best_iou'] = best_iou
        logs['fill_requirement'] = best_iou >= 0.84
        print(' - val_best_iou: {}  fill_requirement {}'.format(best_iou, best_iou >= 0.84))

# Model

def UNet(img_shape):
    
    def _conv(prev_layer, filters, kernel_size, strides=1):
        conv = Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(prev_layer)
        return conv
        
    def convolution_block(x, filters, size, activation=True):
        x = _conv(x, filters, size)
        x = BatchNormalization()(x)
        if activation == True:
            x = ReLU()(x)
        return x

    def residual_block(blockInput, num_filters=16):
        x = ReLU()(blockInput)
        x = BatchNormalization()(x)
        x = convolution_block(x, num_filters, (3,3) )
        x = _squeeze(x, 16)
        x = convolution_block(x, num_filters, (3,3), activation=False)
        x = Add()([x, blockInput])
        return x

    def _decoder_block(prev_layer, dim, padding):
        main_branch = Conv2DTranspose(dim, 3, strides=2, padding=padding, kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(prev_layer)
        return main_branch

    def _encoder_block(prev_layer, dim):
        x = _conv(prev_layer, dim, (3, 3))
        for _ in range(2):
                x = residual_block(x, dim)
        x = ReLU()(x)
        return x

    def _level_block(encoder, dim, depth):
        print(depth, encoder.shape)
        if depth > 0:
            encoder = _encoder_block(encoder, dim)

            downsampler = MaxPooling2D((2, 2))(encoder)
            downsampler = Dropout(0.25)(downsampler)

            level = _level_block(downsampler, int(dim*2), depth-1)

            if encoder.shape[1] in [25, 101]:
                padding = 'valid'
            else:
                padding = 'same'
            decoder = _decoder_block(level, dim, padding)
            decoder = Concatenate()([encoder, decoder])
            decoder = Dropout(0.5)(decoder)

            decoder = _encoder_block(decoder, dim)
        else:
            decoder = _encoder_block(encoder, dim)
        return decoder

    def _shortcut(x, residual):
        input_shape = K.int_shape(x)
        residual_shape = K.int_shape(residual)
        stride_width = int(input_shape[2] // residual_shape[2])
        stride_height = int(input_shape[1] // residual_shape[1])
        equal_channes = input_shape[3] == residual_shape[3]
        shortcut = x
        shortcut = Conv2D(filters=residual_shape[3], kernel_size=1, strides=(stride_width, stride_height), padding='same',
            kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)
        addition = Add()([shortcut, residual])

        return ReLU()(addition) 

    def _squeeze(prev_layer, reduction_ratio=4):
        prev_layer_shape = K.int_shape(prev_layer)
        se = GlobalMaxPooling2D()(prev_layer)
        se = Dense(prev_layer_shape[3]//reduction_ratio, activation='relu')(se)
        se = Dense(prev_layer_shape[3], activation='sigmoid')(se)
        return Multiply()([prev_layer, se])

    eps = 1.1e-5
    bn_axis = 3

    i = Input(shape=img_shape)

    c = _level_block(i, 32, 5)

    o = Dropout(0.25)(c)
    o = Conv2D(1, 1, activation='sigmoid', padding='same')(o)
    return Model(inputs=i, outputs=o)

def stacking(input_shape, strides):
    i = Input(shape=input_shape)

    c = Conv1D(1, kernel_size=strides, strides=strides,  activation='sigmoid')(i)
    c = Reshape((101, 101))(c)
    m = Model(i, c)
    m.summary()
    return m

def create_model(img_shape):
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))

    model = UNet(img_shape=img_shape)
    #model.compile(optimizer=Adam(), loss=mixed_dice_bce_loss)
    return model


def fit(model, X_train, Y_train, x_valid, y_valid, output_name, finetune=False, loss_str='mixed'):

    if finetune:
        lr = 0.00005
    else:
        lr = 0.001

    metric = ValGlobalMetrics(x_valid, y_valid)

    if loss_str == 'mixed':
        loss = mixed_dice_bce_loss
    elif loss_str == 'mixed2':
        loss = mixed_dice_bce_loss_masked
    elif loss_str == 'mixed3':
        loss = mixed_bce_lovasz
    elif loss_str == 'lovasz':
        loss = lovasz_loss

    model.compile(optimizer=Adam(lr), loss=mixed_dice_bce_loss)
    early_stopping = EarlyStopping(
        patience=20, verbose=1, monitor='val_best_iou', mode='max')

    checkpointer = ModelCheckpoint(
        output_name, save_best_only=True, verbose=1, monitor='val_best_iou', mode='max')

    reduce_lr = ReduceLROnPlateau(
        factor=0.7, patience=5, verbose=1, monitor='val_best_iou', mode='max')
    csvlog = CSVLogger('{}_log.csv'.format(output_name.split('.')[0]))

    results = model.fit(X_train, Y_train, validation_data=[x_valid, y_valid], batch_size=32, epochs=300,
                        callbacks=[metric, checkpointer, reduce_lr, csvlog, early_stopping], verbose=1, shuffle=True)
    return results


class TTA_ModelWrapper():
    """A simple TTA wrapper for keras computer vision models.
    Args:
        model (keras model): A fitted keras model with a predict method.
    """

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        """Wraps the predict method of the provided model.
        Augments the testdata with horizontal and vertical flips and
        averages the results.
        Args:
            X (numpy array of dim 4): The data to get predictions for.
        """
        p = self.model.predict(X, verbose=1)

        x1 = np.array([np.fliplr(i) for i in X])
        p1 = self.model.predict(x1, verbose=1)
        p += np.array([np.fliplr(i) for i in p1])

        p /= 2
        return p

"""
to test : https://github.com/nicolov/segmentation_keras/blob/master/model.py
to test : https://github.com/ternaus/TernausNet
to test : https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py

"""
