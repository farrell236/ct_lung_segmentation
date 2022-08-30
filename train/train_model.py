import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import SimpleITK as sitk
import tensorflow as tf

from DeeplabV3Plus import DeeplabV3Plus


# Training Parameters
epochs = 20
batch_size = 8
buffer_size = 1000
learning_rate = 1e-4

# Set LUNA16 Dataset directory
root_dir = '/vol/biodata/data/chest_ct/LUNA16/'
images_df = pd.read_csv('kfold_split.csv')

# Remove missing images
missing_images = [
    '1.3.6.1.4.1.14519.5.2.1.6279.6001.946129570505893110165820050204',
    '1.3.6.1.4.1.14519.5.2.1.6279.6001.771741891125176943862272696845',
    '1.3.6.1.4.1.14519.5.2.1.6279.6001.927394449308471452920270961822',
    '1.3.6.1.4.1.14519.5.2.1.6279.6001.855232435861303786204450738044']
images_df = images_df[images_df.seriesuid.isin(missing_images) == False]


def load_itk_volume(filename):

    image_itk = sitk.ReadImage(root_dir+'images/'+bytes.decode(filename.numpy(), 'utf-8')+'.mhd')
    image_itk = sitk.IntensityWindowing(image_itk, -450., 50)
    image_arr = sitk.GetArrayFromImage(image_itk).astype('uint8')
    image_arr = tf.image.convert_image_dtype(image_arr, tf.float32)[..., None]

    label_itk = sitk.ReadImage(root_dir+'seg-lungs-LUNA16/'+bytes.decode(filename.numpy(), 'utf-8')+'.mhd')
    label_arr = sitk.GetArrayFromImage(label_itk).astype('uint8')
    label_arr = tf.one_hot(label_arr, depth=6, dtype=tf.uint8)

    return image_arr, label_arr


def ensure_shape(image, label):
    image = tf.ensure_shape(image, (None, None, None))
    label = tf.ensure_shape(label, (None, None, None))
    return image, label


# Code adapted from "Generalized dice loss for multi-class segmentation"
# https://github.com/keras-team/keras/issues/9395#issuecomment-370971561
def dice_coef(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient function for LUNA16; ignores background pixel labels 0,1,2
    Pass to model as metric during compile statement
    '''
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true[..., 3:], tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred[..., 3:])
    intersect = tf.keras.backend.sum(y_true_f * y_pred_f, axis=-1)
    denom = tf.keras.backend.sum(y_true_f + y_pred_f, axis=-1)
    return tf.keras.backend.mean((2. * intersect / (denom + smooth)))


def dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef(y_true, y_pred)


for i in range(9):
    print(f'Training on Fold: {i+1}')

    train_uid = images_df[images_df['fold'] != i]['seriesuid']
    valid_uid = images_df[images_df['fold'] == i]['seriesuid']

    train_dataset = tf.data.Dataset.from_tensor_slices(train_uid)
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.map(map_func=lambda x: tf.py_function(
        func=load_itk_volume, inp=[x], Tout=[tf.float32, tf.uint8]), num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
    train_dataset = train_dataset.map(map_func=ensure_shape, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_uid)
    valid_dataset = valid_dataset.map(map_func=lambda x: tf.py_function(
        func=load_itk_volume, inp=[x], Tout=[tf.float32, tf.uint8]), num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
    valid_dataset = valid_dataset.map(map_func=ensure_shape, num_parallel_calls=tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(batch_size)

    # Define DeepLabV3+ Model
    model = DeeplabV3Plus(image_size=(512, 512, 1), num_classes=6)

    # Train with Binary Crossentropy Loss
    csv_logger = tf.keras.callbacks.CSVLogger(f'logs/training_bce_{i}.log')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoints/DeeplabV3Plus_{i}.tf',
        monitor='val_loss', verbose=1,
        save_best_only=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=[checkpoint, csv_logger])

    # Fine tune with Dice Loss (optional)
    csv_logger = tf.keras.callbacks.CSVLogger(f'logs/training_dice_{i}.log')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'checkpoints/DeeplabV3Plus_{i}.tf',
        monitor='val_dice_coef', mode='max', verbose=1,
        save_best_only=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=dice_coef_loss,
        metrics=[dice_coef])
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        callbacks=[checkpoint, csv_logger])

    del model
    del train_dataset
    del valid_dataset
