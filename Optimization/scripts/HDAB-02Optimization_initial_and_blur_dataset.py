from modules import functions_HDAB as func
import pandas as pd
import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.model_selection import train_test_split

input_dir = ['/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_HDAB/',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/blur']

target_dir = ['/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels/',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/blur']

imgs, masks = func.loadDataset(input_dir, target_dir)

target_img_shape = [128, 128, 3]
target_mask_shape = [128, 128, 1]

X, y = func.preProcessDataset(imgs, masks, target_img_shape, target_mask_shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)

losses = ['binary_crossentropy', func.DiceLoss, func.IoULoss, func.Weighted_BCEnDice_loss, tfa.losses.SigmoidFocalCrossEntropy()]
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
        unet = func.UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=1)
        unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = k), 
                     loss=i, 
                     metrics=[tfa.metrics.F1Score(num_classes=2, threshold=j, average='micro'), tf.keras.metrics.AUC(), 
                              tf.keras.metrics.TruePositives(thresholds=j), tf.keras.metrics.FalsePositives(thresholds=j), 
                              tf.keras.metrics.TrueNegatives(thresholds=j), tf.keras.metrics.FalseNegatives(thresholds=j)])
        results = unet.fit(X_train, y_train, batch_size=32, epochs=e, validation_data=(X_val, y_val), verbose=1)
        loss, f1_score, auc, tp, fp, tn, fn = unet.evaluate(X_test, y_test)
        IoU_keras, acc_keras, recall_keras, precision_keras, f1_keras = func.metrics(X_test, y_test, unet, j)
        
        if (i == 'binary_crossentropy'):
          data.append(['Adam', 'BCE', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])
        elif (i == func.DiceLoss):
          data.append(['Adam', 'Dice', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])
        elif (i == func.IoULoss):
          data.append(['Adam', 'IoU', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])
        elif (i == func.Weighted_BCEnDice_loss):
          data.append(['Adam', 'Weighted BCE/Dice', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])
        else:
          data.append(['Adam', 'SFCE', j, k, e, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])

        if f1_score > prev_f1_value:
          cnn_spec = f'Best f1-score get from Adam w Learning rate: {k} - Loss: {i} - Threshold: {j} - Epochs: {e}'
          prev_f1_value = f1_score
          func.saveImgs(X_test, y_test, unet, j, 2)
          unet.save('/home/calvina/saved_models/HDAB/HER2_image_HDAB_segmentation_initial_and_blur_dataset.hdf5')

        func.clear()

  

print(cnn_spec)
################################################################

df = pd.DataFrame(data,
                  columns=['Optimizer', 'Loss Function', 'Threshold', 'Learning rate', 'Epochs', 
                           'Loss', 'Acc', 'Precision', 'Recall', 'F1-Score', 'Mean IoU', 'AUC', 'TP', 'FP', 'TN', 'FN'])

res = df.set_index(['Optimizer', 'Loss Function', 'Threshold', 'Learning rate', 'Epochs'])
print(res)
df.to_csv('/home/calvina/data/HDAB/02optimization_HDAB_initial_and_blur_dataset.csv', index=False)