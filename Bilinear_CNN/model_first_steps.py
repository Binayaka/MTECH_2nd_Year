"""this will be used to run the first steps, to prevent OOM """

import os
import keras

import pre_process
from model_with_loss import build_model, train_model, save_histogram

NO_CLASS = 44

SAVE_PATH = './models/save/'

def save_model(model):
    """this will save the model """
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    model_path = SAVE_PATH + 'model.h5'
    print('Saving model')
    keras.models.save_model(model, model_path, overwrite=True, include_optimizer=True)


def run():
    """first steps """
    gen_dir_train, gen_dir_valid = pre_process.load_data(path_data_train='./images/splitted/train', path_data_valid='./images/splitted/valid', size_mini_batch=10)
    model = build_model(no_class=NO_CLASS, no_last_layer_backbone=17, rate_learning=1.0, rate_decay_weight=1e-8, flg_debug=True)
    train_model(model=model, gen_dir_train=gen_dir_train, gen_dir_valid=gen_dir_valid, max_epoch=1, batch_size=9)
    ### finetune only fc layer
    print('Finetuning FC Layer')
    hist = train_model(model=model, gen_dir_train=gen_dir_train, gen_dir_valid=gen_dir_valid, max_epoch=99)
    print('Finetuning all layers')
    ### finetune all layers
    for layer in model.layers:
        layer.trainable = True
    # now that all layers are trainable, change the LR
    opt_sgd = keras.optimizers.sgd(lr=1e-3, decay=1e-9, momentum=0.9, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer=opt_sgd, metrics=["categorical_accuracy"])
    save_model(model)
    save_histogram(hist, './model/save/histograms/')



if __name__ == "__main__":
    run()
