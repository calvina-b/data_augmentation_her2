from modules import functions_SFCE as func
import pandas as pd
import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.model_selection import train_test_split

input_dir = ['/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_HDAB/',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/blur',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/brightness_down',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/brightness_up',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/contrast',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/crop_50%',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/crop_80%',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/flipped_left_right',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/flipped_up_down',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/noise',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/rotation_90°',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/imgs_augmented_HDAB/transpose']

target_dir = ['/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels/',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/blur',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/brightness_down',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/brightness_up',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/contrast',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/crop_50%',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/crop_80%',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/flipped_left_right',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/flipped_up_down',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/noise',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/rotation_90°',
            '/home/calvina/Jupyter/HER2_images_pixel_segmentation/labels_augmented/transpose']

imgs, masks = func.loadDataset(input_dir, target_dir)

target_img_shape = [128, 128, 3]
target_mask_shape = [128, 128, 1]

X, y = func.preProcessDataset(imgs, masks, target_img_shape, target_mask_shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=123)


prev_f1_value = 0
cnn_spec = ''
data = []

gamma = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
alpha = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00] 


for i in gamma:
  for j in alpha:
    print(f'Training with gamma: {i} and alpha {j}')
    unet = func.UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=1)
    unet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=j, gamma=i), 
                  metrics=[tfa.metrics.F1Score(num_classes=2, threshold=0.4, average='micro'), tf.keras.metrics.AUC(), 
                          tf.keras.metrics.TruePositives(thresholds=0.4), tf.keras.metrics.FalsePositives(thresholds=0.4), 
                          tf.keras.metrics.TrueNegatives(thresholds=0.4), tf.keras.metrics.FalseNegatives(thresholds=0.4)])
    results = unet.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_val, y_val), verbose=1)
    loss, f1_score, auc, tp, fp, tn, fn = unet.evaluate(X_test, y_test)
    IoU_keras, acc_keras, recall_keras, precision_keras, f1_keras = func.metrics(X_test, y_test, unet, 0.4)
    
    data.append(['Adam', 'SFCE', i, j, 0.4, 0.001, 50, loss, acc_keras, precision_keras, recall_keras, f1_score, IoU_keras, auc, tp, fp, tn, fn])

    if f1_score > prev_f1_value:
      cnn_spec = f'Best f1-score get with gamma: {i} - alpha: {j}'
      prev_f1_value = f1_score
      func.saveImgs(X_test, y_test, unet, 0.4, 2)
      unet.save('/home/calvina/saved_models/SFCE/HER2_image_segmentation_HDAB_SFCE_full_dataset.hdf5')

    func.clear()
  

print(cnn_spec)
################################################################

df = pd.DataFrame(data,
                  columns=['Optimizer', 'Loss Function', 'Gamma', 'Alpha', 'Threshold', 'Learning rate', 'Epochs', 
                           'Loss', 'Acc', 'Precision', 'Recall', 'F1-Score', 'Mean IoU', 'AUC', 'TP', 'FP', 'TN', 'FN'])

res = df.set_index(['Optimizer', 'Loss Function', 'Gamma', 'Alpha', 'Threshold', 'Learning rate', 'Epochs'])
print(res)
df.to_csv('/home/calvina/data/SFCE/11optimization_HDAB_SFCE_full_dataset.csv', index=False)