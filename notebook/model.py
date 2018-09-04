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
        print(' - val_best_iou: {}'.format(best_iou))

# Model
def acti_layer(prev_layer, activation):
    if activation == 'relu':
        x = ReLU()(prev_layer)
    elif activation == 'lrelu':
        x = LeakyReLU()(prev_layer)
    elif activation == 'prelu':
        x = PReLU()(prev_layer)
    return x

class Scale(Layer):
    """Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:
        out = in * gamma + beta,
    where 'gamma' and 'beta' are the weights and biases larned.
    Keyword arguments:
    axis -- integer, axis along which to normalize in mode 0. For instance,
        if your input tensor has shape (samples, channels, rows, cols),
        set axis to 1 to normalize per feature map (channels axis).
    momentum -- momentum in the computation of the exponential average 
        of the mean and standard deviation of the data, for 
        feature-wise normalization.
    weights -- Initialization weights.
        List of 2 Numpy arrays, with shapes:
        `[(input_shape,), (input_shape,)]`
    beta_init -- name of initialization function for shift parameter 
        (see [initializers](../initializers.md)), or alternatively,
        Theano/TensorFlow function to use for weights initialization.
        This parameter is only relevant if you don't pass a `weights` argument.
    gamma_init -- name of initialization function for scale parameter (see
        [initializers](../initializers.md)), or alternatively,
        Theano/TensorFlow function to use for weights initialization.
        This parameter is only relevant if you don't pass a `weights` argument.
        
    """
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def UNet(img_shape, start_ch=64, depth=4, inc_rate=2):
    def _conv_block(m, dim):
        eps = 1.1e-5
        bn_axis = 3

        n = Conv2D(dim, 3, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(m)
        n = acti_layer(n, 'relu')
        n = BatchNormalization(epsilon=eps, axis=bn_axis)(n)
        n = Scale(axis=bn_axis)(n)
        n = Dropout(0.5)(n)
        n = Conv2D(dim, 3, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(n)
        n = BatchNormalization(epsilon=eps, axis=bn_axis)(n)
        n = Scale(axis=bn_axis)(n)
        n = acti_layer(n, 'relu')
        se = GlobalMaxPooling2D()(n)
        se = Dense(dim//2, activation='relu')(se)
        se = Dense(dim, activation='sigmoid')(se)
        n = Multiply()([n, se])

        shortcut = Conv2D(dim, 1, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(m)
        shortcut = BatchNormalization(epsilon=eps, axis=bn_axis)(shortcut)
        shortcut = Scale(axis=bn_axis)(shortcut)

        x = Add()([n, shortcut])
        n = acti_layer(x, 'relu')
        
        return n

    def _level_block(m, dim, depth, inc):
        if depth > 0:
            n = _conv_block(m, dim)
            m = MaxPooling2D()(n)
            m = _level_block(m, int(inc*dim), depth-1, inc)

            m = Conv2DTranspose(dim, 3, strides=2, activation=None, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(m)
            m = acti_layer(m, 'relu')

            n = Concatenate()([n, m])
            m = _conv_block(n, dim)
        else:
            m = _conv_block(m, dim)
        return m

    eps = 1.1e-5
    bn_axis = 3

    i = Input(shape=img_shape)
    x = ZeroPadding2D((2,2))(i)
    x = Conv2D(start_ch, (3, 3), strides=(1,1), padding='same', name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    
    o = _level_block(x, start_ch, depth, inc_rate)
    o = Cropping2D(cropping=(2,2))(o)
    o = Conv2D(1, 1, activation='sigmoid')(o)
    return Model(inputs=i, outputs=o)


def create_model(img_shape, start_ch=32, depth=0, type=0):
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.tensorflow_backend.set_session(tf.Session(config=config))

    model = UNet(img_shape=img_shape, start_ch=start_ch, depth=depth)
    model.compile(optimizer='adam', loss=mixed_dice_bce_loss)
    return model


def fit(model, X_train, Y_train, x_valid, y_valid, output_name):
    early_stopping = EarlyStopping(
        patience=20, verbose=1, monitor='val_best_iou', mode='max')

    checkpointer = ModelCheckpoint(
        output_name, save_best_only=True, verbose=1, monitor='val_best_iou', mode='max')

    reduce_lr = ReduceLROnPlateau(
        factor=0.5, patience=5, min_lr=1e-6, verbose=1, monitor='val_best_iou', mode='max')
    csvlog = CSVLogger('{}_log.csv'.format(output_name.split('.')[0]))

    results = model.fit(X_train, Y_train, validation_data=[x_valid, y_valid], batch_size=32, epochs=200,
                        callbacks=[ValGlobalMetrics(x_valid, y_valid), checkpointer, reduce_lr, csvlog, early_stopping], verbose=1, shuffle=True)
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

        x1 = np.array([np.flipud(i) for i in X])
        p1 = self.model.predict(x1, verbose=1)
        p += np.array([np.flipud(i) for i in p1])

        x1 = np.array([np.fliplr(i) for i in x1])
        p1 = self.model.predict(x1, verbose=1)
        p += np.array([np.fliplr(np.flipud(i)) for i in p1])

        p /= 4
        return p

"""
to test : https://github.com/nicolov/segmentation_keras/blob/master/model.py
to test : https://github.com/ternaus/TernausNet
to test : https://github.com/neptune-ml/open-solution-salt-detection/blob/master/src/unet_models.py

"""
