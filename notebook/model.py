
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, PReLU, ReLU
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
import numpy as np

dice_weight = K.variable(0.0)

class MyCallback(Callback):
    def __init__(self, dice_weight):
        self.dice_weight = dice_weight

    def on_epoch_end(self, epoch, logs={}):
        K.set_value(self.dice_weight, min(5.0, K.get_value(self.dice_weight) + 0.2))
        print('Dice weight {:0.2f}'.format(K.get_value(self.dice_weight)))

def multiclass_dice_loss(output, target):
    return 1 - (2* K.sum(output*target)) / (K.sum(output) + K.sum(target) + 1e-7)

def mixed_dice_bce_loss(y_true, y_pred):
    bce_loss = binary_crossentropy(y_true=y_true, y_pred=y_pred)
    dice_loss = multiclass_dice_loss(y_pred, y_true)
    return 5*dice_loss+ bce_loss

def green_unit(filter):

    def f(i):
        x = Conv2D(filter, (3, 3), activation=None, padding='same',kernel_regularizer=regularizers.l2(0.0001))(i)
        x = BatchNormalization() (x)
        x = PReLU()(x)
        x = Dropout(0.1)(x)
        x = Conv2D(filter, (3, 3), activation=None, padding='same',kernel_regularizer=regularizers.l2(0.0001))(x)
        x = BatchNormalization() (x)
        x = PReLU()(x)
        x = add([i,x])
        return x

    return f

def red_unit(filter):

    def f(i):
        x = Conv2D(filter, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (i)
        x = BatchNormalization() (x)
        x = PReLU()(x)
        x = Dropout(0.1)(x)
        x = Conv2D(filter, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
        x = BatchNormalization() (x)
        x = PReLU()(x)
        x = MaxPooling2D()(x)
        return x
    return f

def yellow_unit(filter):
    def f(i):
        x = Conv2DTranspose(filter, (2, 2), strides=(2, 2), padding='same') (i)
        x = PReLU()(x)
        x = Conv2D(filter, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
        x = BatchNormalization() (x)
        x = PReLU()(x)
        x = Conv2D(filter, (3, 3), activation=None, padding='same', kernel_regularizer=regularizers.l2(0.0001)) (x)
        x = BatchNormalization() (x)
        x = PReLU()(x)
        return x
    return f

def create_model(im_height, im_width, im_chan):
    inputs = Input((im_height, im_width, im_chan))
    s = Lambda(lambda x: x/255)(inputs)

    # Green unit
    x_0_0 = green_unit(8*2)(s)
    x_1_0 = red_unit(16*2)(x_0_0)
    x_2_0 = red_unit(32*2)(x_1_0)
    x_3_0 = red_unit(64*2)(x_2_0)
    x_4_0 = red_unit(128*2)(x_3_0)

    print(x_0_0.shape)
    print(x_1_0.shape)
    print(x_2_0.shape)
    print(x_3_0.shape)
    print(x_4_0.shape)

    # Decoder
    x_4_1 = green_unit(128*2)(x_4_0)
    x_3_1 = concatenate([yellow_unit(64*2)(x_4_1), green_unit(64*2)(x_3_0)])
    x_2_1 = concatenate([yellow_unit(32*2)(x_3_1), green_unit(32*2)(x_2_0)])
    x_1_1 = concatenate([yellow_unit(16*2)(x_2_1), green_unit(16*2)(x_1_0)])
    x_0_1 = concatenate([yellow_unit(8*2)(x_1_1), green_unit(8*2)(x_0_0)])
    x_0_1 = green_unit(8*2*2)(x_0_1)
    print('decoder')
    print(x_4_1.shape)
    print(x_3_1.shape)
    print(x_2_1.shape)
    print(x_1_1.shape)
    print(x_0_1.shape)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (x_0_1)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=mixed_dice_bce_loss)
    return model


def fit(model, X_train, Y_train, x_valid, y_valid, output_name):
    early_stopping = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint(output_name, save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-5, verbose=1)
    csvlog = CSVLogger('{}_log.csv'.format(output_name.split('.')[0]))

    results = model.fit(X_train, Y_train, validation_data=[x_valid, y_valid], batch_size=32, epochs=200,
                        callbacks=[checkpointer, reduce_lr, csvlog, early_stopping], verbose=1, shuffle=True)
    return results

"""
to test : https://github.com/nicolov/segmentation_keras/blob/master/model.py
to test : https://github.com/ternaus/TernausNet
to test : https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py

"""
