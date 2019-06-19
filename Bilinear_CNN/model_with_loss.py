"""This will be the model used """
import os
import random
import time
import numpy as np
import tensorflow as tf
import keras
import keras.backend
import keras.layers
import keras.applications
from keras.initializers import glorot_normal
from keras.utils.vis_utils import plot_model

import pre_process

## seeding
os.environ['PYTHONHASHSEED'] = '3'
np.random.seed(3)
random.seed(3)
tf.set_random_seed(3)

## memory allocation
CONFIG = tf.ConfigProto()
CONFIG.gpu_options.allow_growth = True
#CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.5
SESSION = tf.Session(config=CONFIG)
keras.backend.set_session(SESSION)

### number of output classes, 44 for MalayaKew
NO_CLASS = 44


def outer_product(x):
    """calculate outer product of two tensors """
    return keras.backend.batch_dot(x[0], x[1], axes=[1, 1]) / x[0].get_shape().as_list()[1]


def signed_sqrt(x):
    """calculate element-wise signed square root of a tensor"""
    return keras.backend.sign(x) * keras.backend.sqrt(keras.backend.abs(x) + 1e-9)

def l2_norm(x, axis=-1):
    """calculate l2-norm """
    return keras.backend.l2_normalize(x, axis=axis)

def build_model(size_height=448,
                size_weight=448,
                no_class=44,
                no_last_layer_backbone=17,
                name_optimizer="sgd",
                rate_learning=1.0,
                rate_decay_learning=0.0,
                rate_decay_weight=0.0,
                name_initializer="glorot_normal",
                name_activation_logits="softmax",
                name_loss="categorical_crossentropy",
                flg_debug=False,
                **kwargs):
    """this will build the model """
    keras.backend.clear_session()
    print('-------------------------------------------')
    print('parameters:')
    for key, val in locals().items():
        if not val is None and not key == 'kwargs':
            print('\t', key, '=', val)
    print('-------------------------------------------')
    ### load pre-trained model
    tensor_input = keras.layers.Input(shape=[size_height, size_weight, 3])
    model_detector = keras.applications.vgg16.VGG16(input_tensor=tensor_input, include_top=False, weights='imagenet')
    #bi-linear pooling
    x_detector = model_detector.layers[no_last_layer_backbone].output
    shape_detector = model_detector.layers[no_last_layer_backbone].output_shape
    if flg_debug:
        print('shape_detector: {}'.format(shape_detector))
    #extract features from extractor, same with detector for symmetry DxD model
    shape_extractor = shape_detector
    x_extractor = x_detector
    if flg_debug:
        print('shape_extractor: {}'.format(shape_extractor))
    #reshape to (minibatch_size, total_pixels, filter_size)
    x_detector = keras.layers.Reshape([shape_detector[1] * shape_detector[2], shape_detector[-1]])(x_detector)
    if flg_debug:
        print('x_detector after reshape ops: {}'.format(x_detector.shape))
    x_extractor = keras.layers.Reshape([shape_extractor[1] * shape_extractor[2], shape_extractor[-1]])(x_extractor)
    if flg_debug:
        print('x_extractor after reshape ops: {}'.format(x_extractor.shape))
    #outer product of features
    x = keras.layers.Lambda(outer_product)([x_detector, x_extractor])
    if flg_debug:
        print('x shape after outer product ops: {}'.format(x.shape))
    #reshape to (minibatch_size, filtersize_detector * filtersize_extractor)
    x = keras.layers.Reshape([shape_detector[-1] * shape_extractor[-1]])(x)
    if flg_debug:
        print('x shape after reshape ops: {}'.format(x.shape))
    #signed sqrt
    x = keras.layers.Lambda(signed_sqrt)(x)
    if flg_debug:
        print('x shape after signed square root ops: {}'.format(x.shape))
    #l2 normalization
    x = keras.layers.Lambda(l2_norm)(x)
    if flg_debug:
        print('x shape after l2 normalization ops: {}'.format(x.shape))

    ###attach FC layer
    if name_initializer is not None:
        name_initializer = eval(name_initializer+"()")
    else:
        name_initializer = glorot_normal()
    x = keras.layers.Dense(units=no_class,
                           kernel_regularizer=keras.regularizers.l2(rate_decay_weight),
                           kernel_initializer=name_initializer)(x)
    if flg_debug:
        print('x shape after dense ops: {}'.format(x.shape))
    tensor_prediction = keras.layers.Activation(name_activation_logits)(x)
    if flg_debug:
        print('prediction shape: {}'.format(tensor_prediction.shape))
    ###
    ###compile model
    ###
    model_bilinear = keras.models.Model(inputs=[tensor_input], outputs=[tensor_prediction])
    #fix pre-trained weights
    for layer in model_detector.layers:
        layer.trainable = False
    #define optimizers
    opt_adam = keras.optimizers.adam(lr=rate_learning, decay=rate_decay_learning)
    opt_rms = keras.optimizers.rmsprop(lr=rate_learning, decay=rate_decay_learning)
    opt_sgd = keras.optimizers.sgd(lr=rate_learning, decay=rate_decay_learning, momentum=0.9, nesterov=False)
    optimizers = {"adam" : opt_adam, "rmsprop" : opt_rms, "sgd" : opt_sgd}
    model_bilinear.compile(loss=name_loss, optimizer=optimizers[name_optimizer], metrics=["categorical_accuracy"])
    if flg_debug:
        print(model_bilinear.summary())
    return model_bilinear

def train_model(model=None, name_model='BCNN_Keras', gen_dir_train=None, gen_dir_valid=None, max_epoch=50, batch_size=11):
    """this will train the model """
    path_model = './model/{}/'.format(name_model)
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    #callback_setting
    callback_logger = keras.callbacks.CSVLogger(path_model + 'log_training_{}.csv'.format(now), separator=',', append=False)
    callback_saver = keras.callbacks.ModelCheckpoint(path_model +
                                                     "E[{epoch:02d}]" +
                                                     "_LOS[{loss:.3f}]" +
                                                     "_ACC[{categorical_accuracy:.3f}]" +
                                                     ".hdf5",
                                                     monitor='loss',
                                                     verbose=0,
                                                     mode='auto',
                                                     period=10,
                                                     save_best_only=True)
    # val_loss is not available since we dont pass validation generator, saving on loss
    callback_reducer = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, min_delta=1e-3)
    callback_stopper = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-3, patience=10, verbose=0, mode='auto')
    list_callbacks = [callback_logger, callback_saver, callback_reducer, callback_stopper]
    ### Important, it seems that due to race conditions, workers must be set to 1, else it may get stuck on last iteration
    ### it seems there are several problems with fit_generator, trying a different solution
    # validation_step_size = len(gen_dir_valid) // batch_size
    # print('validation_step_size::\t', validation_step_size)
    # it seems there may be a problem with the validation generator
    #hist = model.fit_generator(gen_dir_train, epochs=max_epoch, validation_data=gen_dir_valid, callbacks=list_callbacks, workers=3, verbose=1)
    train_step_size = len(gen_dir_train) // batch_size
    hist = model.fit_generator(gen_dir_train, steps_per_epoch=train_step_size, epochs=max_epoch, callbacks=list_callbacks, workers=3, verbose=1)
    model.save_weights(path_model
                       + "E[{}]".format(len(hist.history['loss']))
                       + "_LOS[{:.3f}]".format(hist.history['loss'][-1])
                       + "_ACC[{:.3f}]".format(hist.history['categorical_accuracy'][-1])
                       + ".h5")
    return hist

def save_histogram(hist, path):
    """this will save the histogram to the given folder """
    now = time.strftime("%Y%m%d%H%M%S", time.localtime())
    path = path + str(now) + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    val_loss = path + "_loss.txt"
    val_cat_acc = path + "_cat_acc.txt"
    #np.array(hist.history['loss']).dump(open(str(val_loss), 'w'))
    #np.array(hist.history['categorical_accuracy']).dump(open(str(val_cat_acc), 'w'))
    loss_array = np.array(hist.history['loss'])
    acc_array = np.array(hist.history['categorical_accuracy'])
    print(repr(loss_array))
    print(repr(acc_array))
    np.savetxt(val_loss, loss_array)
    np.savetxt(val_cat_acc, acc_array)


def runner():
    """actual method for running the model """
    ###
    ### pre-trained model specification, using VGG16 "block5_conv3", translates to layer number 17
    ###
    gen_dir_train, gen_dir_valid = pre_process.load_data(path_data_train='./images/splitted/train', path_data_valid='./images/splitted/valid', size_mini_batch=11)
    model = build_model(no_class=NO_CLASS, no_last_layer_backbone=17, rate_learning=1.0, rate_decay_weight=1e-8, flg_debug=True)
    try:
        plot_model(model, to_file='model_plot.png', show_layer_names=True, show_shapes=True)
        plot_model(model, to_file='model_plot.gv', show_layer_names=True, show_shapes=True)
        plot_model(model, to_file='model_plot.svg', show_layer_names=True, show_shapes=True)
    except OSError as identifier:
        print(identifier)
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
    print('Begin final approach')
    hist = train_model(model=model, gen_dir_train=gen_dir_train, gen_dir_valid=gen_dir_valid, max_epoch=1)
    print('Approaching final training')
    hist = train_model(model=model, gen_dir_train=gen_dir_train, gen_dir_valid=gen_dir_valid, max_epoch=33)
    print('Done!, save histogram')
    save_histogram(hist, './model/BCNN_Keras/histograms/')


if __name__ == '__main__':
    runner()
