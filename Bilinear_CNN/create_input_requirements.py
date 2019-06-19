"""This will create the files needed for the bilinear cnn """

import os
import shutil
import numpy as np
import PIL
import PIL.Image
from tqdm import tqdm

TRAIN_PATH = '../data/train/Class'
TEST_PATH = '../data/test/Class'

IMAGES_PATH = './files/images.txt'
IMAGES_CLASS_LABELS_PATH = './files/image_class_labels.txt'
TRAIN_TEST_SPLIT_PATH = './files/train_test_split.txt'

def create_all_files():
    """This will create the list for the images file, the images_class_label file as well as the split file"""
    print('Creating the files')
    index = 1
    with open(IMAGES_PATH, 'w+') as images, open(IMAGES_CLASS_LABELS_PATH, 'w+') as labels, open(TRAIN_TEST_SPLIT_PATH, 'w+') as split_file:
        for i in tqdm(range(1, 45)):
            # iterate over both folders, read their names, and add to the list
            train_path = TRAIN_PATH + str(i)
            test_path = TEST_PATH  + str(i)
            #print('Train path :: {0}, test path::{1}'.format(train_path, test_path))
            for path in os.listdir(train_path):
                link = os.path.abspath(train_path + '/' + path)
                string = str(index) + ' ' + link + '\n'
                lbl_str = str(index) + ' ' + str(i) + '\n'
                split_str = str(index) + ' 1 \n'
                index = index + 1
                string = string.replace('\\', '/')
                images.write(string)
                labels.write(lbl_str)
                split_file.write(split_str)
            for path in os.listdir(test_path):
                link = os.path.abspath(test_path + '/' + path)
                string = str(index) + ' ' + link + '\n'
                lbl_str = str(index) + ' ' + str(i) + '\n'
                split_str = str(index) + ' 0 \n'
                index = index + 1
                string = string.replace('\\', '/')
                images.write(string)
                labels.write(lbl_str)
                split_file.write(split_str)
    print('\nDone with all files')

def create_directory(path):
    """This will create the directory if it doesn't exist """
    if not os.path.exists(path):
        os.makedirs(path)

def create_child_directories(path):
    """This will be used to create sub directories """
    string = path + '/Class'
    for i in range(1, 45):
        curr_str = string + str(i)
        create_directory(curr_str)

def setup_folders(path):
    """This will setup the input folders """
    create_directory(path)
    create_child_directories(path)

def get_required_path(path):
    """This will extract the required path from the given path """
    first_slash = path.rfind('/')
    second_slash = path.rfind('/', 0, first_slash-1)
    return path[second_slash:]


def create_directories(n_sample=0):
    """This will create the train test split directories """
    print('Creating train and test directories')
    path_data = './images'
    path_data_train = path_data + '/splitted/train'
    path_data_valid = path_data + '/splitted/valid'
    setup_folders(path_data_train)
    setup_folders(path_data_valid)
    list_path_file = np.genfromtxt('./files/images.txt', dtype=str)
    list_flg_split = np.genfromtxt('./files/train_test_split.txt', dtype=np.uint8)
    list_mean_train = np.zeros(3)
    list_std_train = np.zeros(3)
    list_sample = []
    count_train = 0
    count_valid = 0
    for i in tqdm(range(len(list_path_file))):
        path_file = list_path_file[i, 1]
        image = PIL.Image.open(path_file)
        image_np = np.array(image)
        image.close()
        file_name = get_required_path(path_file)
        if count_train + count_valid < n_sample:
            list_sample.append(image_np)
        if list_flg_split[i, 1] == 1:
            count_train += 1
            # put this file in the valid location
            copy_path = path_data_train + file_name
            if not os.path.exists(copy_path):
                shutil.copy(path_file, copy_path)
            for dim in range(3):
                list_mean_train[dim] += image_np[:, :, dim].mean()
                list_std_train[dim] += image_np[:, :, dim].std()
        else:
            count_valid += 1
            # put this file in the valid location
            copy_path = path_data_valid + file_name
            if not os.path.exists(copy_path):
                shutil.copy(path_file, copy_path)
    list_mean_train /= count_train
    list_std_train /= count_train
    print("N of train:\t", count_train)
    print("N of valid:\t", count_valid)
    print("mean of train:\t", list_mean_train)
    print("std of train:\t", list_std_train)
    return np.asarray(list_sample)





if __name__ == '__main__':
    create_all_files()
    TRAINVAL = create_directories(n_sample=10)
    np.array(TRAINVAL).dump(open('dump.numpy', 'wb'))
    print('Dumped data to dump.numpy')
