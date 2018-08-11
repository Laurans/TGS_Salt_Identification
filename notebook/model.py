import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, PReLU
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import numpy as np

def create_model(im_height, im_width, im_chan):
    inputs = Input((im_height, im_width, im_chan))
    s = Lambda(lambda x: x/255)(inputs)

    b0 = BatchNormalization()(s)
    c0 = Conv2D(8, (3, 3), activation=None, padding='same') (b0)
    c0 = PReLU()(c0)
    c0 = Conv2D(8, (3, 3), activation=None, padding='same') (c0)
    c0 = PReLU()(c0)
    p0 = MaxPooling2D((2, 2)) (c0)

    b1 = BatchNormalization()(p0)
    c1 = Conv2D(16, (3, 3), activation=None, padding='same') (b1)
    c1 = PReLU()(c1)
    c1 = Conv2D(16, (3, 3), activation=None, padding='same') (c1)
    c1 = PReLU()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    #p1 = Dropout(0.25)(p1)

    b2 = BatchNormalization()(p1)
    c2 = Conv2D(32, (3, 3), activation=None, padding='same') (b2)
    c2 = PReLU()(c2)
    c2 = Conv2D(32, (3, 3), activation=None, padding='same') (c2)
    c2 = PReLU()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    #p2 = Dropout(0.5)(p2)

    b3 = BatchNormalization()(p2)
    c3 = Conv2D(64, (3, 3), activation=None, padding='same') (b3)
    c3 = PReLU()(c3)
    c3 = Conv2D(64, (3, 3), activation=None, padding='same') (c3)
    c3 = PReLU()(c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    #p3 = Dropout(0.5)(p3)

    b4 = BatchNormalization()(p3)
    c4 = Conv2D(128, (3, 3), activation=None, padding='same') (b4)
    c4 = PReLU()(c4)
    c4 = Conv2D(128, (3, 3), activation=None, padding='same') (c4)
    c4 = PReLU()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    #p4 = Dropout(0.5)(p4)

    b5 = BatchNormalization()(p4)
    c5 = Conv2D(256, (3, 3), activation=None, padding='same') (b5)
    c5 = PReLU()(c5)
    c5 = Conv2D(256, (3, 3), activation=None, padding='same') (c5)
    c5 = PReLU()(c5)

    b6 = BatchNormalization()(c5)
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (b6)
    u6 = concatenate([u6, c4])
    #u6 = Dropout(0.5)(u6)
    c6 = Conv2D(128, (3, 3), activation=None, padding='same') (u6)
    c6 = PReLU()(c6)
    c6 = Conv2D(128, (3, 3), activation=None, padding='same') (c6)
    c6 = PReLU()(c6)

    b7 = BatchNormalization()(c6)
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (b7)
    u7 = concatenate([u7, c3])
    #u7 = Dropout(0.5)(u7)
    c7 = Conv2D(64, (3, 3), activation=None, padding='same') (u7)
    c7 = PReLU()(c7)
    c7 = Conv2D(64, (3, 3), activation=None, padding='same') (c7)
    c7 = PReLU()(c7)

    b8 = BatchNormalization()(c7)
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (b8)
    u8 = concatenate([u8, c2])
    #u8 = Dropout(0.5)(u8)
    c8 = Conv2D(32, (3, 3), activation=None, padding='same') (u8)
    c8 = PReLU()(c8)
    c8 = Conv2D(32, (3, 3), activation=None, padding='same') (c8)
    c8 = PReLU()(c8)

    b9 = BatchNormalization()(c8)
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (b9)
    u9 = concatenate([u9, c1], axis=3)
    #u9 = Dropout(0.5)(u9)
    c9 = Conv2D(16, (3, 3), activation=None, padding='same') (u9)
    c9 = PReLU()(c9)
    c9 = Conv2D(16, (3, 3), activation=None, padding='same') (c9)
    c9 = PReLU()(c9)

    b10 = BatchNormalization()(c9)
    u10 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (b10)
    u10 = concatenate([u10, c0], axis=3)
    #u9 = Dropout(0.5)(u9)
    c10 = Conv2D(8, (3, 3), activation=None, padding='same') (u10)
    c10 = PReLU()(c10)
    c10 = Conv2D(8, (3, 3), activation=None, padding='same') (c10)
    c10 = PReLU()(c10)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c10)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def fit(model, X_train, Y_train, x_valid, y_valid, output_name):
    #earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint('{}.h5'.format(output_name), verbose=0, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=0)

    results = model.fit(X_train, Y_train, validation_data=[x_valid, y_valid], batch_size=128, epochs=100,
                        callbacks=[checkpointer, reduce_lr], verbose=1)
    return results
