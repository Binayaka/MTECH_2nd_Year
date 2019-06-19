"""this is for the second step """
import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras


import pre_process
from model_with_loss import train_model, save_histogram

NO_CLASS = 44

SAVE_PATH = './models/save/'

def save_model(model):
    """this will save the model """
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    model_path = SAVE_PATH + 'model.h5'
    print('Saving model')
    keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)

def load_existing_model(path):
    """this will load the existing model """
    model = keras.models.load_model(path, custom_objects={'keras': keras})
    print('Loaded model from disk')
    return model


def run():
    """this will run the second step """
    gen_dir_train, gen_dir_valid = pre_process.load_data(path_data_train='./images/splitted/train', path_data_valid='./images/splitted/valid', size_mini_batch=10)
    model = load_existing_model(SAVE_PATH + 'model.h5')
    opt_sgd = keras.optimizers.sgd(lr=1e-3, decay=1e-9, momentum=0.9, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=opt_sgd, metrics=["categorical_accuracy"])
    print('Begin final approach')
    hist = train_model(model=model, gen_dir_train=gen_dir_train, gen_dir_valid=gen_dir_valid, max_epoch=1)
    print('Approaching final training')
    hist = train_model(model=model, gen_dir_train=gen_dir_train, gen_dir_valid=gen_dir_valid, max_epoch=33)
    print('Done!, save histogram')
    save_model(model)
    save_histogram(hist, './model/save/histograms/')

if __name__ == "__main__":
    keras.backend.clear_session()
    run()
