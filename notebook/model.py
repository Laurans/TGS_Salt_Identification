import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization, PReLU, ReLU
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input

import numpy as np

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

    base_model =  VGG16(include_top=False)

    b1 = BatchNormalization() (s)
    c1 = Conv2D(64, (3, 3), activation=None, padding='same') (b1)
    c1 = ReLU()(c1)
    c1 = Conv2D(64, (3, 3), activation=None, padding='same') (c1)
    c1 = ReLU()(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    b2 = BatchNormalization() (p1)
    c2 = Conv2D(128, (3, 3), activation=None, padding='same') (b2)
    c2 = ReLU()(c2)
    c2 = Conv2D(128, (3, 3), activation=None, padding='same') (c2)
    c2 = ReLU()(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    b3 = BatchNormalization() (p2)
    c3 = Conv2D(256, (3, 3), activation=None, padding='same') (b3)
    c3 = ReLU()(c3)
    c3 = Conv2D(256, (3, 3), activation=None, padding='same') (c3)
    c3 = ReLU()(c3)
    c3 = Conv2D(256, (3, 3), activation=None, padding='same') (c3)
    c3 = ReLU()(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    b4 = BatchNormalization() (p3)
    c4 = Conv2D(512, (3, 3), activation=None, padding='same') (b4)
    c4 = ReLU()(c4)
    c4 = Conv2D(512, (3, 3), activation=None, padding='same') (c4)
    c4 = ReLU()(c4)
    c4 = Conv2D(512, (3, 3), activation=None, padding='same') (c4)
    c4 = ReLU()(c4)
    p4 = MaxPooling2D((2, 2)) (c4)

    b5 = BatchNormalization() (p4)
    c5 = Conv2D(512, (3, 3), activation=None, padding='same') (b5)
    c5 = ReLU()(c5)
    c5 = Conv2D(512, (3, 3), activation=None, padding='same') (c5)
    c5 = ReLU()(c5)
    c5 = Conv2D(512, (3, 3), activation=None, padding='same') (c5)
    c5 = ReLU()(c5)

    b6 = BatchNormalization() (c5)
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (b6)
    u6 = concatenate([u6, c4])
    b6 = BatchNormalization() (u6)
    c6 = Conv2D(512, (3, 3), activation=None, padding='same') (b6)
    c6 = ReLU()(c6)
    c6 = Conv2D(512, (3, 3), activation=None, padding='same') (c6)
    c6 = ReLU()(c6)

    b7 = BatchNormalization() (c6)
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (b7)
    u7 = concatenate([u7, c3])
    b7 = BatchNormalization() (u7)
    c7 = Conv2D(256, (3, 3), activation=None, padding='same') (b7)
    c7 = ReLU()(c7)
    c7 = Conv2D(256, (3, 3), activation=None, padding='same') (c7)
    c7 = ReLU()(c7)

    b8 = BatchNormalization() (c7)
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (b8)
    u8 = concatenate([u8, c2])
    b8 = BatchNormalization() (u8)
    c8 = Conv2D(128, (3, 3), activation=None, padding='same') (b8)
    c8 = ReLU()(c8)
    c8 = Conv2D(128, (3, 3), activation=None, padding='same') (c8)
    c8 = ReLU()(c8)

    b9 = BatchNormalization() (c8)
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (b9)
    u9 = concatenate([u9, c1], axis=3)
    b9 = BatchNormalization() (u9)
    c9 = Conv2D(64, (3, 3), activation=None, padding='same') (b9)
    c9 = ReLU()(c9)
    c9 = Conv2D(64, (3, 3), activation=None, padding='same') (c9)
    c9 = ReLU()(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    return model


def fit(model, X_train, Y_train, x_valid, y_valid, output_name):
    earlystopper = EarlyStopping(patience=5, verbose=1, monitor='val_mean_iou', mode='max')
    checkpointer = ModelCheckpoint(output_name, monitor='val_mean_iou', mode='max', verbose=0, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=0)
    csvlog = CSVLogger('{}_log.csv'.format(output_name.split('.')[0]))

    results = model.fit(X_train, Y_train, validation_data=[x_valid, y_valid], batch_size=64, epochs=100,
                        callbacks=[checkpointer, reduce_lr, earlystopper, csvlog], verbose=1)
    return results

"""
to test : https://github.com/nicolov/segmentation_keras/blob/master/model.py
to test : https://github.com/ternaus/TernausNet
to test : https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py

"""
