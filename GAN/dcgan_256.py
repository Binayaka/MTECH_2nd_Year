"""This is an attempt to use 256px input and output images for applying dc-gan """
import os
import glob
from math import floor
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from keras.layers import Conv2D, LeakyReLU, BatchNormalization, Dense, AveragePooling2D, GaussianNoise
from keras.layers import Reshape, UpSampling2D, Activation, Dropout, Flatten, Conv2DTranspose
from keras.models import model_from_json, Sequential
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session

CLASS_NAME = 'Class1'

INPUT_PATH = './data/train/' + CLASS_NAME + '/'
SAVE_PATH = './data/generated/' + CLASS_NAME + '/'
D_LOSS_REAL = SAVE_PATH + 'discriminator_loss_real.txt'
D_LOSS_FAKE = SAVE_PATH + 'discriminator_loss_fake.txt'
G_LOSS_ALL = SAVE_PATH + 'g_loss_all.txt'
WEIGHTS_PATH = SAVE_PATH + 'models/'
GENERATOR_MODEL_PATH = WEIGHTS_PATH + 'gen.json'
DISCRIMINATOR_MODEL_PATH = WEIGHTS_PATH + 'dis.json'

ITERS = 100000 # default is 500000

def make_path(path):
    """Create the given folder if it doesn't exist """
    if not os.path.exists(path):
        os.makedirs(path)

def create_working_directories():
    """This will setup our working directories. Call it at program startup """
    make_path(SAVE_PATH)
    make_path(WEIGHTS_PATH)

def write_loss(path, loss):
    """This will write the loss to the file """
    with open(path, 'a+') as writer:
        string = str(loss) + '\n'
        writer.write(string)

def write_d_loss_real(loss):
    """This will write the judge real loss"""
    write_loss(D_LOSS_REAL, loss)

def write_d_loss_fake(loss):
    """This will write the judge fake loss """
    write_loss(D_LOSS_FAKE, loss)

def write_g_loss_all(loss):
    """This will write the generator loss """
    write_loss(G_LOSS_ALL, loss)

def zero():
    """This will return an all zeroes array """
    return np.random.uniform(0.0, 0.01, size=[1])

def one():
    """This will return an all ones array """
    return np.random.uniform(0.99, 1.0, size=[1])

def noise(shape):
    """This will return an array of random values """
    return np.random.uniform(-1.0, 1.0, size=[shape, 4096])

def import_images():
    """This will import the images """
    files = glob.glob(INPUT_PATH + '*.jpg')
    images = []
    for image in tqdm(files):
        temp1 = Image.open(image)
        temp = np.array(temp1.convert('RGB'), dtype='float32')
        images.append(temp / 255)
        images.append(np.flip(images[-1], 1))
    return images

class GAN(object):
    """This will be our whole GAN class """
    def __init__(self):
        #Models
        self.D = None
        self.G = None
        self.OD = None
        self.DM = None
        self.AM = None
        #config
        self.LR = 0.0001
        self.steps = 1

    def discriminator(self):
        """This will the discriminator function """
        #if we already have a discriminator, return it
        if self.D:
            return self.D
        self.D = Sequential()

        #add a gaussian noise to prevent Discriminator overfitting
        self.D.add(GaussianNoise(0.2, input_shape=[256, 256, 3]))

        #256x256x3 image
        self.D.add(Conv2D(filters=8, kernel_size=3, padding='same'))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        #128x128x8
        self.D.add(Conv2D(filters=16, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        #64x64x16
        self.D.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        #32x32x32
        self.D.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        #16x16x64
        self.D.add(Conv2D(filters=128, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        #8x8x128
        self.D.add(Conv2D(filters=256, kernel_size=3, padding='same'))
        self.D.add(BatchNormalization(momentum=0.7))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dropout(0.25))
        self.D.add(AveragePooling2D())

        #4x4x256
        self.D.add(Flatten())

        #256
        self.D.add(Dense(128))
        self.D.add(LeakyReLU(0.2))
        self.D.add(Dense(1, activation='sigmoid'))

        return self.D

    def generator(self):
        """This will be the generator function """
        #if we already have a generator, return it
        if self.G:
            return self.G
        self.G = Sequential()
        self.G.add(Reshape(target_shape=[1, 1, 4096], input_shape=[4096]))

        #1x1x4096
        self.G.add(Conv2DTranspose(filters=256, kernel_size=4))
        self.G.add(Activation('relu'))

        #4x4x256 - kernel sizes increased by 1
        self.G.add(Conv2D(filters=256, kernel_size=4, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        #8x8x256 - kernel sizes increased by 1
        self.G.add(Conv2D(filters=128, kernel_size=4, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        #16x16x128
        self.G.add(Conv2D(filters=64, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        #32x32x64
        self.G.add(Conv2D(filters=32, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        #64x64x32
        self.G.add(Conv2D(filters=16, kernel_size=3, padding='same'))
        self.G.add(BatchNormalization(momentum=0.7))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())

        #128x128x16
        self.G.add(Conv2D(filters=8, kernel_size=3, padding='same'))
        self.G.add(Activation('relu'))
        self.G.add(UpSampling2D())
        
        #256x256x8 
        self.G.add(Conv2D(filters=3, kernel_size=3, padding='same'))
        self.G.add(Activation('sigmoid'))

        return self.G

    def dismodel(self):
        """Setup dis model """
        if self.DM is None:
            self.DM = Sequential()
            self.DM.add(self.discriminator())
        learning_rate = self.LR * (0.85 ** floor(self.steps / 10000))
        self.DM.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        return self.DM

    def admodel(self):
        """Setup gen model """
        if self.AM is None:
            self.AM = Sequential()
            self.AM.add(self.generator())
            self.AM.add(self.discriminator())
        learning_rate = self.LR * (0.85 ** floor(self.steps / 10000))
        self.AM.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        return self.AM

    def sod(self):
        """Set discriminator weights """
        self.OD = self.D.get_weights()

    def lod(self):
        """Load Discriminator weights """
        self.D.set_weights(self.OD)

class GanModel(object):
    """This will setup the whole model """
    def __init__(self, images):
        self.images = images
        self.GAN = GAN()
        self.dismodel = self.GAN.dismodel()
        self.admodel = self.GAN.admodel()
        self.generator = self.GAN.generator()

    def save(self, num):
        """This will save the state of the GAN """
        gen_json = self.GAN.G.to_json()
        dis_json = self.GAN.D.to_json()

        with open(GENERATOR_MODEL_PATH, 'w+') as json_file:
            json_file.write(gen_json)

        with open(DISCRIMINATOR_MODEL_PATH, 'w+') as json_file:
            json_file.write(dis_json)

        self.GAN.G.save_weights(WEIGHTS_PATH + 'gen' + str(num) + '.h5')
        self.GAN.D.save_weights(WEIGHTS_PATH + 'dis' + str(num) + '.h5')
        print('Model number {0} saved!', str(num))

    def load(self, num):
        """This will load the state of the GAN with the corresponding weight"""
        steps1 = self.GAN.steps

        #clear the GAN
        self.GAN = None
        self.GAN = GAN()

        #load the generator
        gen_file = open(GENERATOR_MODEL_PATH, 'r')
        gen_json = gen_file.read()
        gen_file.close()

        self.GAN.G = model_from_json(gen_json)
        self.GAN.G.load_weights(WEIGHTS_PATH + 'gen' + str(num) + '.h5')

        #load the discriminator
        dis_file = open(DISCRIMINATOR_MODEL_PATH, 'r')
        dis_json = dis_file.read()
        dis_file.close()

        self.GAN.D = model_from_json(dis_json)
        self.GAN.D.load_weights(WEIGHTS_PATH + 'dis' + str(num) + '.h5')

        #Re-initialize
        self.generator = self.GAN.generator()
        self.dismodel = self.GAN.dismodel()
        self.admodel = self.GAN.admodel()

        self.GAN.steps = steps1

    def evaluate(self):
        """This will show the current generated image """
        im_no = random.randint(0, len(self.images) - 1)
        im1 = self.images[im_no]

        im2 = self.generator.predict(noise(2))

        plt.figure(1)
        plt.imshow(im1)

        plt.figure(2)
        plt.imshow(im2[0])

        plt.figure(3)
        plt.imshow(im2[1])

        plt.show()

    def eval2(self, num=0):
        """This will save the resultant image """
        im2 = self.generator.predict(noise(48))

        first = np.concatenate(im2[:8], axis=1)
        second = np.concatenate(im2[8:16], axis=1)
        third = np.concatenate(im2[16:24], axis=1)
        fourth = np.concatenate(im2[24:32], axis=1)
        fifth = np.concatenate(im2[32:40], axis=1)
        sixth = np.concatenate(im2[40:48], axis=1)

        complete = np.concatenate([first, second, third, fourth, fifth, sixth], axis=0)
        final_image = Image.fromarray(np.uint8(complete * 255))
        image_iter_str = '{:05d}'.format(num)
        final_image.save(SAVE_PATH + image_iter_str + '.png')

    def train_gen(self, batch):
        """This will train the generator """
        self.GAN.sod()
        label_data = []
        for _ in range(int(batch)):
            label_data.append(zero())
        g_loss = self.admodel.train_on_batch(noise(batch), np.array(label_data))
        self.GAN.lod()
        return g_loss

    def train_dis(self, batch):
        """This will train the discriminator """
        #get real images
        im_no = random.randint(0, len(self.images) - batch - 1)
        train_data = self.images[im_no : im_no + int(batch/2)]
        label_data = []
        for _ in range(int(batch/2)):
            label_data.append(zero())
        d_loss_real = self.dismodel.train_on_batch(np.array(train_data), np.array(label_data))

        #get fake images
        train_data = self.generator.predict(noise(int(batch/2)))
        label_data = []
        for _ in range(int(batch/2)):
            label_data.append(one())
        d_loss_fake = self.dismodel.train_on_batch(train_data, np.array(label_data))
        return (d_loss_real, d_loss_fake)

    def train(self, batch=16):
        """This is the actual training method """
        (real_loss, fake_loss) = self.train_dis(batch)
        gen_loss = self.train_gen(batch)
        write_d_loss_fake(fake_loss)
        write_d_loss_real(real_loss)
        write_g_loss_all(gen_loss)

        if self.GAN.steps % 500 == 0:
            self.save(floor(self.GAN.steps / 1000))
            #self.evaluate()

        if self.GAN.steps % 5000 == 0:
            self.GAN.AM = None
            self.GAN.DM = None
            self.admodel = self.GAN.admodel()
            self.dismodel = self.GAN.dismodel()
        self.GAN.steps = self.GAN.steps + 1

if __name__ == '__main__':
    CONFIG = tf.ConfigProto()
    CONFIG.gpu_options.allow_growth = True
    SESS = tf.Session(config=CONFIG)
    set_session(SESS)
    create_working_directories()
    IMAGES = import_images()
    MODEL = GanModel(IMAGES)
    MODEL.GAN.D.summary()
    MODEL.GAN.G.summary()

    print('Beginning training, this is going to take a while')
    for _ in tqdm(range(ITERS)):
        MODEL.train()
        if MODEL.GAN.steps % 500 == 0:
            #print('\nRound ', str(MODEL.GAN.steps))
            MODEL.eval2(int(MODEL.GAN.steps / 500))
    print('\nDone!\n')
