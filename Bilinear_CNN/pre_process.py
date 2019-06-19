"""This will be used for any pre-processing required """
import os
import numpy as np
import keras
import keras.backend as backend
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import image_utilities as util


class DirectoryIterator(keras.preprocessing.image.DirectoryIterator):
    """This class is our implementation of the default directory iterator for keras """
    def _get_batches_of_transformed_samples(self, index_array):
        """This will be used for sampling """
        batch_x = np.zeros((len(index_array), ) + self.image_shape, dtype=backend.floatx())
        grayscale = self.color_mode == 'grayscale'

        #build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=None,
                           interpolation=self.interpolation)
            x = img_to_array(img, data_format=self.data_format)
            #pillow images should be closed after 'load_img', but not pil images
            if hasattr(img, 'close'):
                img.close()
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

        #optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix, index=j, hash=np.random.randint(1e7), format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        #build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()
        elif self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(backend.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_classes), dtype=backend.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


class ImageDataGenerator(keras.preprocessing.image.ImageDataGenerator):
    """This is our implementation of the default image data generator """
    def flow_from_directory(self, 
                            directory,
                            target_size=(256, 256),
                            color_mode='rgb',
                            classes=None,
                            class_mode='categorical',
                            batch_size=16,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix='',
                            save_format='png',
                            follow_links=False,
                            subset=None,
                            interpolation='nearest'):
        """returns our directory iterator """
        return DirectoryIterator(directory, self, target_size=target_size, color_mode=color_mode,
                                 classes=classes, class_mode=class_mode,
                                 data_format=self.data_format,
                                 batch_size=batch_size, shuffle=shuffle,
                                 seed=seed, save_to_dir=save_to_dir,
                                 save_prefix=save_prefix, save_format=save_format,
                                 follow_links=follow_links, subset=subset,
                                 interpolation=interpolation)


def load_data(path_data_train=None, path_data_valid=None, size_width=448, size_height=448, size_mini_batch=16):
    """For loading a small set of data """
    #set pre-processing functions
    func_train = lambda x: util.normalize_image(util.random_crop_image(util.horizontal_flip_image(util.resize_image(x, size_target=(size_height, size_width), flg_keep_aspect=True))),
                                                mean=[82.61621345, 106.17492596, 61.47832594])
    func_valid = lambda x: util.normalize_image(util.center_crop_image(util.horizontal_flip_image(util.resize_image(x, size_target=(size_height, size_width), flg_keep_aspect=True))),
                                                mean=[82.61621345, 106.17492596, 61.47832594])
    #set image data generator
    gen_train = ImageDataGenerator(preprocessing_function=func_train)
    gen_valid = ImageDataGenerator(preprocessing_function=func_valid)
    gen_dir_train = gen_train.flow_from_directory(path_data_train, target_size=(size_height, size_width), batch_size=size_mini_batch)
    gen_dir_valid = gen_valid.flow_from_directory(path_data_valid, target_size=(size_height, size_width), batch_size=size_mini_batch, shuffle=False)
    return gen_dir_train, gen_dir_valid


if __name__ == '__main__':
    GEN_DIR_TRAIN, GEN_DIR_VALID = load_data(path_data_train='./images/splitted/train', path_data_valid='./images/splitted/valid', size_mini_batch=11)
    X_TRAIN, Y_TRAIN = GEN_DIR_TRAIN.next()
    np.array(X_TRAIN).dump(open('x_train_preprocess', 'wb'))
    np.array(Y_TRAIN).dump(open('y_train_preprocess', 'wb'))
    print('Done with pre-processing')
