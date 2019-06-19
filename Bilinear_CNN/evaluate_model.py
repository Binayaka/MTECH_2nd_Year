"""this will be used to evaluate the test images """
from matplotlib import pyplot as plt
import numpy as np
import keras
from keras.preprocessing import image
import pandas as pd
import image_utilities as util
from pre_process import ImageDataGenerator

SAVE_PATH = './models/save/'

def load_existing_model(path):
    """this will load the existing model """
    model = keras.models.load_model(path, custom_objects={'keras': keras})
    print('Loaded model from disk')
    return model

def test_image():
    """this will test an image """
    path = './images/splitted/valid/Class1/Class1(1)R135_00023.jpg'
    img = image.load_img(path, target_size=(448, 448))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255

    plt.imshow(img_tensor[0])
    plt.show()
    print(img_tensor.shape)

def load_generators(path_train=None, path_test=None, size_width=448, size_height=448):
    """this will return the train and the test generators """
    func_train = lambda x: util.normalize_image(util.random_crop_image(util.horizontal_flip_image(util.resize_image(x, size_target=(size_height, size_width), flg_keep_aspect=True))),
                                                mean=[82.61621345, 106.17492596, 61.47832594])
    func_test = lambda x: util.normalize_image(util.center_crop_image(util.horizontal_flip_image(util.resize_image(x, size_target=(size_height, size_width), flg_keep_aspect=True))),
                                               mean=[82.61621345, 106.17492596, 61.47832594])
    gen_train = ImageDataGenerator(preprocessing_function=func_train)
    gen_test = ImageDataGenerator(preprocessing_function=func_test)
    gen_dir_train = gen_train.flow_from_directory(path_train, target_size=(size_height, size_width), batch_size=9)
    gen_dir_test = gen_test.flow_from_directory(path_test, target_size=(size_height, size_width), batch_size=10, shuffle=False)
    return gen_dir_train, gen_dir_test

def evaluate_model():
    """this will evaluate the model """
    model = load_existing_model(SAVE_PATH + 'model.h5')
    gen_dir_train, gen_dir_test = load_generators(path_train='./images/splitted/train', path_test='./images/test/')
    labels = (gen_dir_train.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    print(labels)
    #model.evaluate_generator(generator=gen_dir_test, verbose=1)
    pred = model.predict_generator(generator=gen_dir_test, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    predictions = [labels[k] for k in predicted_class_indices]
    filenames = gen_dir_test.filenames
    results = pd.DataFrame({"Filename":filenames, "Predictions":predictions})
    results.to_csv("results.csv", index=False)
    print('Done')

if __name__ == "__main__":
    evaluate_model()
