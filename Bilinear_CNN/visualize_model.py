"""this will be used to visualize the model """
import os
import keras
from keras.preprocessing.image import img_to_array
import cv2
import keract
import numpy as np

SAVE_PATH = './models/save/'
IMG_PATH = './images/test/Class1/Class1(1)R135_00023.jpg'

def makedirs(directory):
    """this will ensure the directory is present """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_existing_model(path):
    """this will load the existing model """
    model = keras.models.load_model(path, custom_objects={'keras': keras})
    print('Loaded model from disk')
    return model

def process_activations(activations, cmap=None, save=False, directory=''):
    """
    Plot heatmaps of activations for all filters for each layer
    :param activations: dict mapping layers to corresponding activations (1, output_h, output_w, num_filters)
    :param cmap: string - a valid matplotlib colourmap to be used
    :param save: bool- if the plot should be saved
    :param directory: target directory, if desired
    :return: None
    """
    import matplotlib.pyplot as plt
    import math
    for layer_name, acts in activations.items():
        print(layer_name, acts.shape, end=' ')
        if acts.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue
        if len(acts.shape) <= 2:
            print('-> Skipped. 2D Activations.')
            continue
        print('')
        nrows = int(math.sqrt(acts.shape[-1]) - 0.001) + 1  # best square fit for the given number
        ncols = int(math.ceil(acts.shape[-1] / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(64, 64))
        fig.suptitle(layer_name)
        for i in range(nrows * ncols):
            if i < acts.shape[-1]:
                if len(acts.shape) == 4:
                    img = acts[0, :, :, i]
                    hmap = axes.flat[i].imshow(img, cmap=cmap)
            axes.flat[i].axis('off')
        fig.subplots_adjust(right=0.8)
        cbar = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(hmap, cax=cbar)
        if save:
            if directory != '':
                makedirs(directory)
            name = directory + '/' + layer_name.split('/')[0] + '.png'
            plt.savefig(name, bbox_inches='tight')
        else:
            plt.show()
        # pyplot figures require manual closing
        plt.close(fig)

def process_heatmaps(activations, input_image, save=False, directory=''):
    """
    Plot heatmaps of activations for all filters overlayed on the input image for each layer
    :param activations: dict mapping layers to corresponding activations (1, output_h, output_w, num_filters)
    :param input_image: input image for the overlay
    :param save: bool- if the plot should be saved
    :return: None
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import math
    for layer_name, acts in activations.items():
        print(layer_name, acts.shape, end=' ')
        if acts.shape[0] != 1:
            print('-> Skipped. First dimension is not 1.')
            continue
        if len(acts.shape) <= 2:
            print('-> Skipped. 2D Activations.')
            continue
        print('')
        nrows = int(math.sqrt(acts.shape[-1]) - 0.001) + 1  # best square fit for the given number
        ncols = int(math.ceil(acts.shape[-1] / nrows))
        fig, axes = plt.subplots(nrows, ncols, figsize=(64, 64))
        fig.suptitle(layer_name)
        # computes values required to scale the activations (which will form our heat map) to be in range 0-1
        scaler = MinMaxScaler()
        scaler.fit(acts.reshape(-1, 1))
        for i in range(nrows * ncols):
            if (i < acts.shape[-1]) and (len(acts.shape) == 4):
                img = acts[0, :, :, i]
                # scale the activations (which will form our heat map) to be in range 0-1
                img = scaler.transform(img)
                # resize heatmap to be same dimensions of input_image
                img = Image.fromarray(img)
                img = img.resize((input_image.shape[0], input_image.shape[1]), Image.BILINEAR)
                img = np.array(img)
                axes.flat[i].imshow(input_image / 255.0)
                # overlay a 70% transparent heat map onto the image
                # Lowest activations are dark, highest are dark red, mid are yellow
                axes.flat[i].imshow(img, alpha=0.3, cmap='jet', interpolation='bilinear')
            axes.flat[i].axis('off')
        if save:
            if directory != '':
                makedirs(directory)
            name = directory + '/' + layer_name.split('/')[0] + '.png'
            plt.savefig(name, bbox_inches='tight')
        else:
            plt.show()
        plt.close(fig)

def visualize_activations(model=None, path=None):
    """this will be used to visualize the activations of the model for the given image """
    image = cv2.imread(path)
    image = cv2.resize(image, (448, 448))
    print('Given Image dimensions: ', image.shape)
    img = img_to_array(image)
    img = img.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    activations = keract.get_activations(model, img)
    process_activations(activations, cmap='viridis', save=True, directory='./activations/')
    print('Done!')

def visualize_heatmaps(model=None, path=None):
    """this will be used to visualize the heatmaps of the model for the given image """
    image = cv2.imread(path)
    image = cv2.resize(image, (448, 448))
    print('Given Image dimensions: ', image.shape)
    img = img_to_array(image)
    arr_image = np.array(img)
    img = img.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    activations = keract.get_activations(model, img)
    process_heatmaps(activations, arr_image, save=True, directory='./heatmaps/')

if __name__ == "__main__":
    MODEL = load_existing_model(SAVE_PATH + 'model.h5')
    OPT_SGD = keras.optimizers.sgd(lr=1e-3, decay=1e-9, momentum=0.9, nesterov=False)
    MODEL.compile(loss="categorical_crossentropy", optimizer=OPT_SGD, metrics=["categorical_accuracy"])
    visualize_heatmaps(model=MODEL, path=IMG_PATH)
