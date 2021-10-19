######################################################################
## Medical segmentation using CNN
## - Loss and metrics (dice, cross-entropy)
##
## Nov 16. 2018
## Youngwon (youngwon08@gmail.com)
##
## Reference
## - Keras (https://github.com/keras-team/keras)
######################################################################

import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from medpy import metric as medmetric

import tensorflow as tf
import keras.backend as K

from keras.losses import mean_squared_error, binary_crossentropy
from keras.metrics import binary_accuracy
#############################################################################################################################
def match_penalty_loss(y_true, y_pred):
    return y_pred[0]

# def weighted_binary_crossentropy(y_true, y_pred):
#     y_pred_clip = K.clip(y_pred, 1e-10, 1-1e-10)
#     w = K.sum(1-y_pred_clip) / K.sum(y_pred_clip)
#     return -K.mean(w*y_true*K.log(y_pred_clip) + (1-y_true)*K.log(1-y_pred_clip))

def precision(y_true, y_pred):
    ytp = y_true*y_pred
    yfp = (1-y_true)*y_pred
    return tf.reduce_mean( tf.reduce_sum(ytp, axis=(1,2,3,4)) / ( tf.reduce_sum(ytp, axis=(1,2,3,4)) + tf.reduce_sum(yfp, axis=(1,2,3,4)) + K.epsilon() ) )

def recall(y_true, y_pred):
    ytp = y_true*y_pred
    yfn = y_true*(1-y_pred)
    return tf.reduce_mean( K.sum(ytp, axis=(1,2,3,4))/( tf.reduce_sum(ytp, axis=(1,2,3,4)) + tf.reduce_sum(yfn, axis=(1,2,3,4)) + K.epsilon() ) )

def dice(y_true, y_pred):
    y_pred_clip = K.clip(y_pred, 1e-10, 1-1e-10)
    y_true_clip = K.clip(y_true, 1e-10, 1-1e-10)
    ytp = y_true_clip * y_pred_clip
    return tf.reduce_mean(2 * K.sum(ytp, axis=(1,2,3,4))/ (K.sum(y_true_clip, axis=(1,2,3,4)) + K.sum(y_pred_clip, axis=(1,2,3,4)) + K.epsilon()))

def negative_dice(y_true, y_pred):
    return -dice(y_true, y_pred)


# TODO: fix

def sensitivity(y_true, y_pred):
#     y_pred = K.round(y_pred)
#     neg_y_pred = 1 - y_pred
#     true_positive = K.sum(y_true * y_pred)
#     false_negative = K.sum(y_true * neg_y_pred)
#     return (true_positive) / (true_positive + false_negative + K.epsilon())
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), 0.5), K.floatx())
    neg_y_pred = 1 - y_pred
    true_positive = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
    false_negative = K.round(K.sum(K.clip(y_true * neg_y_pred, 0, 1)))
    return (true_positive) / (true_positive + false_negative + K.epsilon())

def specificity(y_true, y_pred):
#     y_pred = K.round(y_pred)
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     false_positive = K.sum(neg_y_true * y_pred)
#     true_negative = K.sum(neg_y_true * neg_y_pred)
#     return (true_negative) / (false_positive + true_negative + K.epsilon())
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), 0.5), K.floatx())
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    false_positive = K.round(K.sum(K.clip(neg_y_true * y_pred, 0, 1)))
    true_negative = K.round(K.sum(K.clip(neg_y_true * neg_y_pred, 0, 1)))
    return (true_negative) / (false_positive + true_negative + K.epsilon())

def negative_minimum_sensitivity_specificity(y_true, y_pred):
    neg_y_pred = 1 - y_pred
    true_positive = K.sum(y_true * y_pred)
    false_negative = K.sum(y_true * neg_y_pred)
    smooth_sensitivity =  (true_positive) / (true_positive + false_negative + K.epsilon())

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    false_positive = K.sum(neg_y_true * y_pred)
    true_negative = K.sum(neg_y_true * neg_y_pred)
    smooth_specificity = (true_negative) / (false_positive + true_negative + K.epsilon())
    
    return -K.minimum(smooth_sensitivity, smooth_specificity)

def distance(a,b):
    """
    Euclidean distance
    Need to correct with true distence weight (from image offset)
    """
    return K.sqrt(K.sum(K.square(a-b)))
    
def where3D(a):
    a_shape = K.shape(a)
    z = K.repeat(K.arange(a_shape[0]), a_shape[1]*a_shape[2]).reshape(a_shape[0],a_shape[1],a_shape[2])
    x = K.repeat(K.array([K.repeat(K.arange(a_shape[1]), a_shape[2]).reshape(a_shape[1],a_shape[2])]),
                  a_shape[0], axis=0)
    y = K.repeat(K.array([K.repeat(K.array([K.arange(a_shape[2])]), a_shape[1], axis=0)]), a_shape[0], axis=0)
    return K.stack([z[a], x[a], y[a]]).transpose()
    
def asd(A_in, B_in):
    """
    Average surface distance (ASD)
    """
    # TODO : fix (to be surface version)
    A = K.cast(where3D(K.not_equal(A_in, 0)), 'float32')
    B = K.cast(where3D(K.not_equal(B_in, 0)), 'float32')
    def dist_fn(x): return K.min(K.map_fn(lambda y: distance(x[0],y), x[1]))
    return K.mean(K.map_fn(lambda x: dist_fn((x,B)), A))
    

def assd(y_true, y_pred):
    """Average symmetric surface distance (ASSD)"""
    return (asd(y_true,y_pred)+asd(y_pred,y_true)) * 0.5
    
def assd_round(y_true, y_pred_in):
    """Average symmetric surface distance (ASSD)"""
    y_pred = K.round(y_pred_in)
    return (asd(y_true,y_pred)+asd(y_pred,y_true)) * 0.5
    
# def hd_nonsym(A_in, B_in):
    # A = K.cast(where3D(K.not_equal(A_in, 0)), 'float32')
    # B = K.cast(where3D(K.not_equal(B_in, 0)), 'float32')
    # def dist_fn(x): return K.min(K.map_fn(lambda y: distance(x[0],y), x[1]))
    # return K.max(K.map_fn(lambda x: dist_fn((x,B)), A))

def hd(y_true, y_pred):
    """Hausdorff distance (HD)"""
    return K.maximum(hd_nonsym(y_true, y_pred),hd_nonsym(y_pred,y_true))

def hd_round(y_true, y_pred_in):
    """Hausdorff distance (HD)"""
    y_pred = K.round(y_pred_in)
    return K.maximum(hd_nonsym(y_true, y_pred),hd_nonsym(y_pred,y_true))

# ###############################################################################################
# # loss for smaller labels

def label_center(y_true, y_pred):
    return tf.slice(y_true, (K.shape(y_true)-K.shape(y_pred))/2, K.shape(y_pred))

def binary_crossentropy_sample(y_true, y_pred):
    y_true_sample = label_center(y_true, y_pred)
    return binary_crossentropy(y_true_sample, y_pred)

def weighted_binary_crossentropy_sample(y_true, y_pred):
    y_true_sample = label_center(y_true, y_pred)
    return weighted_binary_crossentropy(y_true_sample, y_pred)

def precision_sample(y_true, y_pred):
    y_true_sample = label_center(y_true, y_pred)
    y_pred_round = K.round(y_pred)
    y_pred_round_clip = K.clip(y_pred_round, 1e-7, 1-1e-7)
    return K.sum(y_true*y_pred_round_clip)/(K.sum(y_true_sample*y_pred_round_clip) + K.sum((1-y_true_sample)*y_pred_round_clip))

def recall_sample(y_true, y_pred):
    y_true_sample = label_center(y_true, y_pred)
    y_pred_round = K.round(y_pred)
    y_pred_round_clip = K.clip(y_pred_round, 1e-7, 1-1e-7)
    return K.sum(y_true*y_pred_round_clip)/(K.sum(y_true_sample*y_pred_round_clip) + K.sum(y_true_sample*(1-y_pred_round_clip)))

def dice_sample(y_true, y_pred):
    y_true_sample = label_center(y_true, y_pred)
    y_pred_round = K.round(y_pred)
    y_pred_round_clip = K.clip(y_pred_round, 1e-7, 1-1e-7)
    return 2 * K.sum(y_true_sample * y_pred_round_clip)/ (K.sum(y_true_sample) + K.sum(y_pred_round_clip))

def negative_dice_sample(y_true, y_pred):
    return -dice_sample(y_true, y_pred)

def assd_sample(y_true, y_pred):
    """Average symmetric surface distance (ASSD)"""
    y_true_sample = label_center(y_true, y_pred)
    y_pred_round = K.round(y_pred)
    return (asd(y_true_sample,y_pred_round)+asd(y_pred_round,y_true_sample)) * 0.5

def hd_sample(y_true, y_pred):
    """Hausdorff distance (HD)"""
    y_true_sample = label_center(y_true, y_pred)
    y_pred_round = K.round(y_pred)
    return K.maximum(hd_nonsym(y_true_sample, y_pred_round),hd_nonsym(y_pred_round,y_true_sample))


###############################################################################################################################
# helper

class PlotLosses(object):
    def __init__(self, figsize=(8,6)):
        plt.plot([], []) 
    
    def __call__(self, nn, train_history):
        train_loss = np.array([i["train_loss"] for i in nn.train_history_])
        valid_loss = np.array([i["valid_loss"] for i in nn.train_history_])

        plt.gca().cla()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(train_loss, label="train loss") 
        plt.plot(valid_loss, label="test loss")

        plt.legend()
        plt.draw()
        
def metric_test_simple(y_true, y_pred):
    # numpy metric_test
    # diceAB = 2 * np.sum(y_true * y_pred)/ (np.sum(y_true) + np.sum(y_pred) + 1e-7)
    # precisionAB = np.sum(y_true*y_pred)/(np.sum(y_true*y_pred) + np.sum((1-y_true)*y_pred) + 1e-7)
    # recallAB = np.sum(y_true*y_pred)/(np.sum(y_true*y_pred) + np.sum(y_true*(1-y_pred)) + 1e-7)
    
    # diceAB = 2 * (np.sum(y_true * y_pred) + 0.5*1e-7)/ (np.sum(y_true) + np.sum(y_pred) + 1e-7)
    # precisionAB = (np.sum(y_true*y_pred) + 1e-7)/(np.sum(y_true*y_pred) + np.sum((1-y_true)*y_pred) + 1e-7)
    # recallAB = (np.sum(y_true*y_pred) + 1e-7)/(np.sum(y_true*y_pred) + np.sum(y_true*(1-y_pred)) + 1e-7)
    
    diceAB = medmetric.dc(y_pred,y_true)
    precisionAB = medmetric.precision(y_pred,y_true)
    recallAB = medmetric.recall(y_pred,y_true)
    return diceAB, precisionAB, recallAB
        
def metric_test(y_true, y_pred, spacing):
    # numpy metric_test
    # diceAB = 2 * (np.sum(y_true * y_pred))/ (np.sum(y_true) + np.sum(y_pred) + 1e-7)
    
    # diceAB = 2 * (np.sum(y_true * y_pred) + 0.5*1e-7)/ (np.sum(y_true) + np.sum(y_pred) + 1e-7)
    # precisionAB = (np.sum(y_true*y_pred) + 1e-7)/(np.sum(y_true*y_pred) + np.sum((1-y_true)*y_pred) + 1e-7)
    # recallAB = (np.sum(y_true*y_pred) + 1e-7)/(np.sum(y_true*y_pred) + np.sum(y_true*(1-y_pred)) + 1e-7)
    
#     A = np.transpose(np.nonzero(y_true)).astype(np.float)
#     B = np.transpose(np.nonzero(y_pred)).astype(np.float)
#     if B.shape[0] == 0:
#         asdA = 0
#         asdB = 0
#         hdA = 0
#         hdB = 0
#     elif A.shape[0] == 0:
#         asdA = 0
#         asdB = 0
#         hdA = 0
#         hdB = 0
#     else:
#         AB = list(map(lambda x: np.min(np.sqrt(np.sum(np.square((x-B)*np.array(spacing)), axis=1))), A))
#         BA = list(map(lambda x: np.min(np.sqrt(np.sum(np.square((x-A)*np.array(spacing)), axis=1))), B))
#         asdA = np.mean(AB)
#         asdB = np.mean(BA)
#         hdA = np.max(AB)
#         hdB = np.max(BA)
    # return diceAB, 0.5*(asdA+asdB), max(hdA, hdB), precisionAB, recallAB
    
    diceAB = medmetric.dc(y_pred,y_true)
    precisionAB = medmetric.precision(y_pred,y_true)
    recallAB = medmetric.recall(y_pred,y_true)
    # asdAB = medmetric.asd(y_pred, y_true, voxelspacing=spacing)
    try:
        assdAB = medmetric.assd(y_pred, y_true, voxelspacing=spacing)
    except:
        assdAB = np.nan
    try:
        hdAB = medmetric.hd(y_pred, y_true, voxelspacing=spacing)
    except:
        hdAB = np.nan
    return diceAB, assdAB, hdAB, precisionAB, recallAB


###############################################################################################################################
        
if __name__ == "__main__":
    print('Test loss functions (Dice / ASSD / HD / precision / recall)')
    y_true_set = np.array([[[0,0,0,0,0],
                            [0,0,0,0,0],
                            [0,1,1,0,0],
                            [1,1,1,0,0],
                            [0,1,0,0,0]]])
    y_pred_set = np.array([[[0,0,0,0,1],
                            [0,0,0,0,0],
                            [0,1,0.6,0,0],
                            [0,1,1,0,0],
                            [0,0.3,0,0,0]]])
    
    def test(acc, y_true_set, y_pred_set):
        sess = tf.Session()
        K.set_session(sess)
        with sess.as_default():
            return acc.eval(feed_dict={y_true: y_true_set, y_pred: y_pred_set})
    
    # tf
    y_true = tf.placeholder("float32", shape=(None,y_true_set.shape[1],y_true_set.shape[2])) 
    y_pred = tf.placeholder("float32", shape=(None,y_pred_set.shape[1],y_pred_set.shape[2]))

    #acc = keras.metrics.binary_crossentropy(y_true, y_pred)
    #acc = sum_binary_crossentropy(y_true, y_pred)
    metric_list = [dice(y_true, y_pred), 
                   assd(y_true, y_pred),
                   hd(y_true, y_pred),
                   precision(y_true, y_pred),
                   recall(y_true, y_pred)]

    # numpy
    print('Dice\t ASSD\t HD\t precision\t recall')
    print('tf : {}'.format([test(acc, y_true_set, y_pred_set) for acc in metric_list]))
    print('np : {}'.format(np.round(metric_test(y_true_set[0],y_pred_set[0]),8)))