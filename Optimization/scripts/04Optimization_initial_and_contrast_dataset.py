import os
import numpy as np
import random
import shutil

import pandas as pd
import skimage
import skimage.io
from skimage.transform import resize
from PIL import Image
import matplotlib.pyplot as plt
import imageio

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda
from tensorflow.keras import Model
from keras import backend as K
from tensorflow.keras.metrics import MeanIoU

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

def loadDataset(path1, path2):
    input_img_paths = []
    target_img_paths = []
    
    if (isinstance(path1, str) == True):
        input_img_paths = sorted(
            [
                os.path.join(path1, fname)
                for fname in os.listdir(path1)
                if fname.endswith(".png")
            ]
        )
    else:
        for i in path1:
            for subdir, dirs, files in os.walk(i):
                for file in files:
                    input_img_paths.append(os.path.join(subdir, file))
        input_img_paths.sort()
        
    
    if (isinstance(path2, str) == True):
        target_img_paths = sorted(
            [
                os.path.join(path2, fname)
                for fname in os.listdir(path2)
                if fname.endswith(".png") and not fname.startswith(".")
            ]
        )
    else:
        for i in path2:
            for subdir, dirs, files in os.walk(i):
                for file in files:
                    target_img_paths.append(os.path.join(subdir, file))
        target_img_paths.sort()
    
    
    return(input_img_paths, target_img_paths)

def preProcessDataset(img, mask, target_imgs_shape, target_masks_shape):
    length = len(img)
    img_height, img_width, img_channels = target_imgs_shape
    mask_height, mask_width, mask_channels = target_masks_shape
    
    X = np.zeros((length, img_height, img_width, img_channels), dtype=np.float32)
    y = np.zeros((length, mask_height, mask_width, mask_channels), dtype=np.int32)
    
    for i in range(length):
        img_ = skimage.io.imread(img[i])[:,:,:img_channels]
        img_ = resize(img_, (img_height, img_width), mode = 'constant', preserve_range = True)
        X[i] = img_ / 255


    for i in range(length):
        mask_ = Image.open(mask[i])
        mask_ = mask_.resize((mask_height, mask_width))
        mask_ = np.reshape(mask_,(mask_height, mask_width, mask_channels))
        y[i] = mask_ / 255
        
    return(X, y)

def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(n_filters, 
                  3,   # Kernel size   
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    
   
    conv = BatchNormalization()(conv, training=False)

    
    if dropout_prob > 0:     
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

   
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv

    
    skip_connection = conv
    
    return next_layer, skip_connection


def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    up = Conv2DTranspose(
                 n_filters,
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')(prev_layer_input)

    merge = concatenate([up, skip_layer_input], axis=3)
    
    
    
    conv = Conv2D(n_filters, 
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv

def UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=1):
    inputs = Input(input_size)
    
    cblock1 = EncoderMiniBlock(inputs, n_filters,dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters * 2,dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters * 4,dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters * 8,dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False) 
    

    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1],  n_filters * 8)
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1],  n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1],  n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1],  n_filters)


    conv9 = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, activation='sigmoid', padding='same')(conv9)
    
    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10, name="U-net")

    return model

def threshold(X_test, unet, value):
    y_pred = unet.predict(X_test)
    y_pred_thresholded = y_pred > value
    
    return(y_pred_thresholded)

def metrics(X_test, y_test, unet, value):
    #IOU
    n_classes = 2
    y_pred_thresholded = threshold(X_test, unet, value)
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(y_pred_thresholded, y_test)

    #ACCURACY
    acc_keras = tf.keras.metrics.Accuracy()
    acc_keras.update_state(y_pred_thresholded, y_test)
    
    #RECALL
    recall_keras = tf.keras.metrics.Recall()
    recall_keras.update_state(y_pred_thresholded, y_test)
    
    #PRECISION
    precision_keras = tf.keras.metrics.Precision()
    precision_keras.update_state(y_pred_thresholded, y_test)
    
    #F1
    f1_keras = (2 * precision_keras.result().numpy() * recall_keras.result().numpy())/(precision_keras.result().numpy() + recall_keras.result().numpy())
    
    return(IOU_keras.result().numpy(), acc_keras.result().numpy(), recall_keras.result().numpy(), precision_keras.result().numpy(), f1_keras)

def saveImgs(X_test, y_test, unet, thresh):
    path = '/home/calvina/imgs'
    if not os.path.exists(path) and not os.path.isdir(path):
        os.mkdir(path)

    path2 = '/home/calvina/imgs/exp4-initial-and-contrast-dataset'
    if os.path.exists(path2) and os.path.isdir(path2):
        shutil.rmtree(path2)
        os.mkdir(path2)
    else:
        os.mkdir(path2)

    for i in range(len(X_test)):
        test_img = X_test[i]
        ground_truth = y_test[i]
        test_img_input = np.expand_dims(test_img, 0)
        prediction = (unet.predict(test_img_input, verbose = 0)[0,:,:,0] > thresh).astype(np.uint8)

        filename_image = 'original_image_%d.png' % i
        filename_mask = 'mask_%d.png' % i
        filename_predict = 'predicted_mask_%d.png' % i
        plt.imsave('%s/%s' % (path2, filename_image), X_test[i])
        plt.imsave('%s/%s' % (path2, filename_mask), ground_truth[:,:,0], cmap='gray')
        plt.imsave('%s/%s' % (path2, filename_predict), prediction, cmap='gray')


def Weighted_BCEnDice_loss(y_true, y_pred):  
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss =  weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight) 
    return loss
    
def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    #logit_y_pred = y_pred
    
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth) # Uptill here is Dice Loss with squared
    loss = 1. - K.sum(score)  #Soft Dice Loss
    return loss

def DiceLoss(y_true, y_pred, smooth=1e-6):   
    # # cast to float32 datatype
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    #flatten label and prediction tensors
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice

def IoULoss(y_true, y_pred, smooth=1e-6):  
    # cast to float32 datatype
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    
    #flatten label and prediction tensors
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(y_true_f * y_pred_f)
    total = K.sum(y_true_f) + K.sum(y_pred_f)
    union = total - intersection
    
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

#############################################################
input_dir = ['/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs/',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented/contrast']

target_dir = ['/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels/',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/contrast']


imgs, masks = loadDataset(input_dir, target_dir)

target_img_shape = [128, 128, 3]
target_mask_shape = [128, 128, 1]

X, y = preProcessDataset(imgs, masks, target_img_shape, target_mask_shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

####################

def clear():
    os.system('clear')

losses = ['binary_crossentropy', DiceLoss, IoULoss, Weighted_BCEnDice_loss, tfa.losses.SigmoidFocalCrossEntropy()]
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
learning_rate = [0.001, 0.002, 0.003]
epochs = [25, 50]
prev_f1_value = 0
cnn_spec = ''
data = []

for i in losses:
  for j in thresholds:
    for k in learning_rate:
      for e in epochs:
        unet = UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=1)
        unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = k), 
                     loss=i, 
                     metrics=[tfa.metrics.F1Score(num_classes=2, threshold=j, average='micro'), tf.keras.metrics.AUC(), 
                              tf.keras.metrics.TruePositives(thresholds=j), tf.keras.metrics.FalsePositives(thresholds=j), 
                              tf.keras.metrics.TrueNegatives(thresholds=j), tf.keras.metrics.FalseNegatives(thresholds=j)])
        results = unet.fit(X_train, y_train, batch_size=32, epochs=e, validation_data=(X_val, y_val), verbose=1)
        loss, f1_score, auc, tp, fp, tn, fn = unet.evaluate(X_test, y_test)
        IoU_keras, acc_keras, recall_keras, precision_keras, f1_keras = metrics(X_test, y_test, unet, j)
        
        if (i == 'binary_crossentropy'):
          data.append(['Adam', 'BCE', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])
        elif (i == DiceLoss):
          data.append(['Adam', 'Dice', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])
        elif (i == IoULoss):
          data.append(['Adam', 'IoU', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])
        elif (i == Weighted_BCEnDice_loss):
          data.append(['Adam', 'Weighted BCE/Dice', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])
        else:
          data.append(['Adam', 'SFCE', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])

        if f1_score > prev_f1_value:
          cnn_spec = f'Best f1-score get from Adam w Learning rate: {k} - Loss: {i} - Threshold: {j} - Epochs: {e}'
          prev_f1_value = f1_score
          saveImgs(X_test, y_test, unet, j)
          unet.save('saved_models/HER2_image_segmentation_initial_and_contrast_dataset.hdf5')

        clear()

  

print(cnn_spec)
################################################################

df = pd.DataFrame(data,
                  columns=['Optimizer', 'Loss Function', 'Threshold', 'Learning rate', 'Epochs', 
                           'Loss', 'Acc', 'Precision', 'Recall', 'F1-Score', 'Mean IoU', 'AUC', 'TP', 'FP', 'TN', 'FN'])

res = df.set_index(['Optimizer', 'Loss Function', 'Threshold', 'Learning rate', 'Epochs'])
print(res)
df.to_csv('04optimization_initial_and_contrast_dataset.csv', index=False)