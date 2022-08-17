import os
import argparse

import numpy as np
import SimpleITK as sitk
import tensorflow as tf

from scipy import ndimage


label_classes = {
    3: 'Left Lung',
    4: 'Right Lung',
    5: 'Trachea'
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Lung Segmentation')
    parser.add_argument('-i', '--input_fn', help='Input CT Lung Volume', required=False)
    parser.add_argument('-o', '--output_fn', help='Output Segmentation Mask')
    parser.add_argument('-m', '--model', help='Trained Model', required=False)
    parser.add_argument('-v', '--verbose', help='Verbose Output', action='store_true', default=False)
    args = vars(parser.parse_args())

    # Load ITK image
    image_itk = image_itk_copy = sitk.ReadImage(args['input_fn'])
    image_itk = sitk.IntensityWindowing(image_itk, -450., 50)
    image_arr = sitk.GetArrayFromImage(image_itk).astype('uint8')
    image_arr = tf.image.convert_image_dtype(image_arr, tf.float32)[..., None]

    # Load trained lung segmentation model
    model = tf.keras.models.load_model(args['model'])

    # Predict segmentation of R-Lung, L-Lung and Trachea
    y_pred = model.predict(image_arr, batch_size=4, verbose=args['verbose'])
    y_pred = np.argmax(y_pred, axis=-1)

    # Post-process and clean predicted segmentation mask
    cleaned_mask = np.zeros_like(y_pred)
    for label in (3, 4, 5):
        if args['verbose']: print(f'Cleaning Label: {label_classes[label]}')
        binary_img = y_pred == label
        label_im, nb_labels = ndimage.label(binary_img)
        sizes = ndimage.sum(binary_img, label_im, range(nb_labels + 1))
        mask = sizes > max(sizes)-1
        cleaned_mask += mask[label_im] * label

    # Copy scan header information to mask
    cleaned_mask = sitk.GetImageFromArray(cleaned_mask.astype('int16'))
    cleaned_mask.CopyInformation(image_itk_copy)

    # Save predicted segmentation mask to disk
    if args['output_fn'] is None:
        filename, extension = os.path.basename(args['input_fn']).split('.', 1)
        fn = filename + '_mask.' + extension
    else:
        fn = args['output_fn']
        
    if args['verbose']: print(f'Saving Mask File: {fn}')
    sitk.WriteImage(cleaned_mask, fn)
