"""This is to test the display method """
import numpy as np
import image_utilities

def run():
    """This is to test the display method """
    print('Running')
    trainval = np.load(open('dump.numpy', 'rb'))
    print('shape of trainval:\t', trainval.shape)
    image_utilities.check_images(trainval)
    #resize images to generate cropped 448 x 448 images, keeping aspect ratio
    print('Resizing')
    trainval_resized = image_utilities.resize_images(trainval, size_target=(448, 448), flg_keep_aspect=True)
    image_utilities.check_images(trainval_resized)
    print('Flipping')
    trainval_resized_flipped = trainval_resized.copy()
    for i in range(9):
        trainval_resized_flipped[i] = image_utilities.horizontal_flip_image(trainval_resized_flipped[i])
    image_utilities.check_images(trainval_resized_flipped)
    print('Center cropping')
    trainval_resized_cropped = trainval_resized.copy()
    for i in range(9):
        trainval_resized_cropped[i] = image_utilities.center_crop_image(trainval_resized_cropped[i])
    image_utilities.check_images(trainval_resized_cropped)
    print('Random cropping')
    trainval_resized_cropped = trainval_resized.copy()
    for i in range(9):
        trainval_resized_cropped[i] = image_utilities.random_crop_image(trainval_resized_cropped[i])
    image_utilities.check_images(trainval_resized_cropped)
    print('Testing pre-processing images')
    preprocessed = np.load(open('x_train_preprocess', 'rb'))
    image_utilities.check_images(preprocessed)

if __name__ == '__main__':
    run()
