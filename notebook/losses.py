from keras import backend as K
import numpy as np
from keras.losses import binary_crossentropy
import tensorflow as tf

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def mixed_dice_bce_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred)+5*dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred)

def focal_loss(y_true, y_pred):
    gamma = 2
    epsilon = 1e-8
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    pred_pt = tf.where(tf.equal(y_true_f, 1), y_pred_f, 1-y_pred_f)
    losses = K.mean(-K.pow(1-pred_pt, gamma)*K.log(pred_pt+epsilon))
    return losses
