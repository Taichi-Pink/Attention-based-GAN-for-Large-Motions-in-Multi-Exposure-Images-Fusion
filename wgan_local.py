import numpy as np
import pickle
import sys, keras
import argparse
import cv2
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge, LeakyReLU
from keras.layers import Reshape, Concatenate
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Flatten
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, Adagrad
from PIL import Image
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import math, imageio, time, os
from utils import *
from data_generate import *
from keras.optimizers import Adam
from random import shuffle
from load_data import *
from glob import glob
from swiss_army_tensorboard import tfboard_loggers
import keras.layers as KL
IN_CH = 3
OUT_CH = 3
# LAMBDA = 100
NF = 64  # number of filter
BATCH_SIZE = 20
num_shots = 3
result_path = 'wgan_local/'
sample_path = 'samples/'
log_path = 'log/'
step_count = 0

##test config
test_path = '/home/ziyi.liu1/dataset/test/'
test_h = 768  # 1024
test_w = 1280  # 1536
batch_size = 1
##train config
train_path = '/home/ziyi.liu1/dataset/tf_records/'
img_rows = 256
img_cols = 256
total_images = 20202
steps = 955

MU = 5000.

tensorboard_2 = keras.callbacks.TensorBoard(
    log_dir=result_path+'log1/',
    batch_size=BATCH_SIZE,
    write_graph=True,
    write_grads=True
)
tensorboard_3 = keras.callbacks.TensorBoard(
    log_dir=result_path+'log2/',
    batch_size=BATCH_SIZE,
    write_graph=True,
    write_grads=True
)
his_log_gen = tfboard_loggers.TFBoardKerasModelWeightsLogger(result_path+"gen/")
his_log_dis = tfboard_loggers.TFBoardKerasModelWeightsLogger(result_path+"dis/")


def tonemap_np(images):  # input/output -1~1
    return np.log(1 + MU * (images + 1) / 2.) / np.log(1 + MU) * 2. - 1


def get_input(scene_dir, image_size):
    c = 3
    in_LDR_paths = sorted(glob(os.path.join(scene_dir, '*aligned.tif')))
    ns = len(in_LDR_paths)

    in_exps_path = os.path.join(scene_dir, 'input_exp.txt')
    in_exps = np.array(open(in_exps_path).read().split('\n')[:ns]).astype(np.float32)
    in_exps -= in_exps.min()
    in_LDRs = np.zeros(image_size + [c * ns], dtype=np.float32)
    in_HDRs = np.zeros(image_size + [c * ns], dtype=np.float32)

    for i, image_path in enumerate(in_LDR_paths):
        img = get_image(image_path, image_size=image_size, is_crop=True)
        in_LDRs[:, :, c * i:c * (i + 1)] = img
        in_HDRs[:, :, c * i:c * (i + 1)] = LDR2HDR(img, 2. ** in_exps[i])

    ref_HDR = get_image(os.path.join(scene_dir, 'ref_hdr_aligned.hdr'), image_size=image_size, is_crop=True)
    return in_LDRs, in_HDRs, ref_HDR
    # return in_LDRs, in_HDRs


# save sample results
def save_results(imgs, out_path):
    imgC = 3
    batchSz, imgH, imgW, c_ = imgs[0].shape
    assert (c_ % imgC == 0)
    ns = c_ // imgC

    nRows = np.ceil(batchSz / 4)
    nCols = min(4, batchSz)  # 4

    res_imgs = np.zeros((batchSz * len(imgs) * ns, imgH, imgW, imgC))
    # rearranging the images, this is a bit complicated
    for n, img in enumerate(imgs):
        for i in range(batchSz):
            for j in range(ns):
                idx = ((i // nCols) * len(imgs) + n) * ns * nCols + (i % nCols) * ns + j
                res_imgs[idx, :, :, :] = img[i, :, :, j * imgC:(j + 1) * imgC]
    save_images(res_imgs, [nRows * len(imgs), nCols * ns], out_path)


########## Additional Helper Functions ##########

def generator_model(test=False, attention=False):
    global BATCH_SIZE
    # imgs: input: 256x256xch
    # U-Net structure, must change to relu
    global img_rows, img_cols
    if test:
        img_rows = test_h
        img_cols = test_w
    inputs = Input((img_rows, img_cols, IN_CH * 3))

    e1 = BatchNormalization()(inputs)
    e1 = Convolution2D(64, 4, 4, subsample=(2, 2), init='uniform', border_mode='same')(e1)
    e1 = LeakyReLU(alpha=0.02)(e1)
    e1 = BatchNormalization()(e1)
    e2 = Convolution2D(128, 4, 4, subsample=(2, 2), init='uniform', border_mode='same')(e1)
    e2 = LeakyReLU(alpha=0.02)(e2)
    e2 = BatchNormalization()(e2)

    ### attention after 2 conv
    if attention:
        e2 = google_attention(e2)
    ###
    e3 = Convolution2D(256, 4, 4, subsample=(2, 2), init='uniform', border_mode='same')(e2)
    e3 = LeakyReLU(alpha=0.02)(e3)
    e3 = BatchNormalization()(e3)
    e4 = Convolution2D(512, 4, 4, subsample=(2, 2), init='uniform', border_mode='same')(e3)
    e4 = LeakyReLU(alpha=0.02)(e4)
    e4 = BatchNormalization()(e4)

    e5 = Convolution2D(512, 4, 4, subsample=(2, 2), init='uniform', border_mode='same')(e4)
    e5 = LeakyReLU(alpha=0.02)(e5)
    e5 = BatchNormalization()(e5)
    e6 = Convolution2D(512, 4, 4, subsample=(2, 2), init='uniform', border_mode='same')(e5)
    e6 = LeakyReLU(alpha=0.02)(e6)
    e6 = BatchNormalization()(e6)

    e7 = Convolution2D(512, 4, 4, subsample=(2, 2), init='uniform', border_mode='same')(e6)
    e7 = LeakyReLU(alpha=0.02)(e7)
    e7 = BatchNormalization()(e7)
    e8 = Convolution2D(512, 4, 4, subsample=(2, 2), init='uniform', border_mode='same')(e7)
    e8 = LeakyReLU(alpha=0.02)(e8)
    e8 = BatchNormalization()(e8)

    d1 = UpSampling2D(interpolation='nearest')(e8)
    d1 = Convolution2D(512, 3, 3, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d1)
    d1 = Concatenate(axis=3)([d1, e7])
    d1 = BatchNormalization()(d1)

    d2 = UpSampling2D(interpolation='nearest')(d1)
    d2 = Convolution2D(512, 3, 3, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d2)
    d2 = Concatenate(axis=3)([d2, e6])
    d2 = BatchNormalization()(d2)

    d3 = Dropout(0.2)(d2)
    d3 = UpSampling2D(interpolation='nearest')(d3)
    d3 = Convolution2D(512, 3, 3, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d3)
    d3 = Concatenate(axis=3)([d3, e5])
    d3 = BatchNormalization()(d3)

    d4 = Dropout(0.2)(d3)
    d4 = UpSampling2D(interpolation='nearest')(d4)
    d4 = Convolution2D(512, 3, 3, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d4)
    d4 = Concatenate(axis=3)([d4, e4])
    d4 = BatchNormalization()(d4)

    d5 = Dropout(0.2)(d4)
    d5 = UpSampling2D(interpolation='nearest')(d5)
    d5 = Convolution2D(256, 3, 3, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d5)
    d5 = Concatenate(axis=3)([d5, e3])
    d5 = BatchNormalization()(d5)

    d6 = Dropout(0.2)(d5)
    d6 = UpSampling2D(interpolation='nearest')(d6)
    d6 = Convolution2D(128, 3, 3, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d6)
    d6 = Concatenate(axis=3)([d6, e2])
    d6 = BatchNormalization()(d6)

    d7 = Dropout(0.2)(d6)
    d7 = UpSampling2D(interpolation='nearest')(d7)
    d7 = Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d7)
    d7 = Concatenate(axis=3)([d7, e1])

    d7 = BatchNormalization()(d7)
    d8 = UpSampling2D(interpolation='nearest')(d7)
    d8 = Convolution2D(3, 3, 3, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d8)
    d8 = BatchNormalization()(d8)
    d9 = Activation('tanh')(d8)

    model = Model(input=inputs, output=d9)
    return model

def activation(input,trainable = True):
    out = KL.Activation('relu',trainable=trainable)(input)
    return out
def discriminator_model(attention=False):
    """ return a (b, 1) logits"""
    model = Sequential()
    model.add(Convolution2D(64, 4, 4, border_mode='same', input_shape=(img_rows, img_cols, IN_CH * 4)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 4, 4, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, 4, 4, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Convolution2D(1, 4, 4, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))

    model.add(Activation('sigmoid'))
    return model
def conv2d(input,filters,kernel_size,strides = 1,trainable = True):
    out = KL.Conv2D(filters = filters,kernel_size = kernel_size,strides = strides,padding = 'same',trainable=trainable)(input)

    out = KL.BatchNormalization(trainable = trainable)(out)
    out = activation(out,trainable=trainable)
    return out

# def global_discriminator(input):
#     out = conv2d(input, filters=64, kernel_size=5, strides=2)
#     out = conv2d(out, filters=128, kernel_size=5, strides=2)
#     out = conv2d(out, filters=256, kernel_size=3, strides=1)
#     out = conv2d(out, filters=512, kernel_size=3, strides=1)
#     out = conv2d(out, filters=256, kernel_size=3, strides=1)
#     out = conv2d(out, filters=128, kernel_size=3, strides=1)
#     out = conv2d(out, filters=1, kernel_size=3, strides=1)
#     # out = KL.Flatten (name = 'gd_flatten 7') (out)
#     # Flatten (), input_shape is said to be unknown and it was not possible to make it flat so cut out the area where the mask bit is set from reshape # x [0] * x [2]: out from the shape and cut out it Set the area to 0)
#     # x [1] * (1 - x [2]): Cut out the region where the bit of mask is not set from input_image
#     # Merge (add) the above two to make the image replaced only with the output of NN for the mask part
#     # out = KL.Reshape((4 * 4 * 512,))(out)
#     # We don apply activation function to output layer
#     # out = KL.Dense(1024)(out)
#     return out
#
#
# def local_discriminator(input):
#     out = conv2d(input, filters=64, kernel_size=3, strides=1)
#     out = conv2d(out, filters=128, kernel_size=3, strides=1)
#     out = conv2d(out, filters=256, kernel_size=3, strides=1)
#     out = conv2d(out, filters=512, kernel_size=3, strides=1)
#     out = conv2d(out, filters=256, kernel_size=3, strides=1)
#     out = conv2d(out, filters=128, kernel_size=3, strides=1)
#     out = conv2d(out, filters=1, kernel_size=3, strides=1)
#     # out = KL.Flatten (name = 'ld_flatten6') (out)
#     # Flatten () Because input_shape was said to be unknown and it could not be flat, we reshaped shape
#     # out = KL.Reshape((2 * 2 * 512,), name='ld_flatten6')(out)
#     # out = KL.Dense(1024)(out)
#     return out

def global_discriminator(input):
    out = conv2d(input, filters=64, kernel_size=5, strides=2)
    out = conv2d(out, filters=128, kernel_size=5, strides=2)

    # out = conv2d(out, filters=256, kernel_size=3, strides=1)
    # out = conv2d(out, filters=512, kernel_size=3, strides=1)
    # out = conv2d(out, filters=256, kernel_size=3, strides=1)
    # out = conv2d(out, filters=128, kernel_size=3, strides=1)
    # out = conv2d(out, filters=1, kernel_size=3, strides=1)
    out = conv2d(out, filters=256, kernel_size=5, strides=2)
    out = conv2d(out, filters=512, kernel_size=5, strides=2)
    out = conv2d(out, filters=512, kernel_size=5, strides=2)
    out = conv2d(out, filters=512, kernel_size=5, strides=2)

    out = KL.Reshape((4 * 4 * 512,))(out)
    out = KL.Dense(1024)(out)
    return out


def local_discriminator(input):
    # out = conv2d(input, filters=64, kernel_size=3, strides=1)
    # out = conv2d(out, filters=128, kernel_size=3, strides=1)
    # out = conv2d(out, filters=256, kernel_size=3, strides=1)
    # out = conv2d(out, filters=512, kernel_size=3, strides=1)
    # out = conv2d(out, filters=256, kernel_size=3, strides=1)
    # out = conv2d(out, filters=128, kernel_size=3, strides=1)
    # out = conv2d(out, filters=1, kernel_size=3, strides=1)

    out = conv2d(input, filters=64, kernel_size=5, strides=2)
    out = conv2d(out, filters=128, kernel_size=5, strides=2)
    out = conv2d(out, filters=256, kernel_size=5, strides=2)
    out = conv2d(out, filters=512, kernel_size=5, strides=2)
    out = conv2d(out, filters=512, kernel_size=5, strides=2)
    out = KL.Reshape((2 * 2 * 512,), name='ld_flatten6')(out)
    out = KL.Dense(1024)(out)
    return out


def api_model( inputs,outputs,trainable = True):
    model = Model(inputs,outputs)
    model.trainable = trainable
    return model
def discriminator_all():
    input_global = Input((img_rows, img_cols, IN_CH * 4))
    input_local = Input((img_rows//4, img_cols//4, IN_CH * 4))
    g_output = global_discriminator(input_global)
    l_output = local_discriminator(input_local)
    out = KL.Lambda(lambda x: K.concatenate(x))([g_output, l_output])
    out = KL.Dense(1)(out)

    model = api_model([input_global, input_local], out)
    # print(model.summary())
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1], 3),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img[:, :, :]
    return image


def crop_random(img, label,x=0.2, y=0.6, N=5):
    imgpatchs = []
    labelpatchs = []
    _,h, w, _= K.int_shape(img)[:]
    size = int(h // 4)

    for i in range(N):
        rand_coe_h = random.random() * (y - x) + x
        rand_coe_w = random.random() * (y - x) + x

        # get width and height of the patch
        rand_h = int(h * rand_coe_h)
        rand_w = int(w * rand_coe_w)

        # the random - generated coordinates are limited in
        # h -> [0, coor_h]
        # w -> [0, coor_w]
        coor_h = h - rand_h
        coor_w = w - rand_w

        # get x and y starting point of the patch
        coor_x = int(random.random() * coor_h)
        coor_y = int(random.random() * coor_w)

        # only create patches for the high layer
        img_patch = img[:,coor_x:coor_x + rand_h, coor_y:coor_y + rand_w,:]
        # resize the patch to [size, size]
        resize_img = tf.image.resize(img_patch, (size, size))
        imgpatchs.append(resize_img)

        # Create patches for the label
        label_patch = label[:,coor_x:coor_x + rand_h, coor_y:coor_y + rand_w,:]
        # resize the patch to [size, size]
        resize_label = tf.image.resize(label_patch, (size, size))
        labelpatchs.append(resize_label)
    labelpatchs_ = tf.convert_to_tensor(labelpatchs)
    imgpatchs_ = tf.convert_to_tensor(imgpatchs)
    return [labelpatchs_, imgpatchs_]

def crop_random0(img, label,real,x=0.2, y=0.6, N=5):
    imgpatchs = []
    labelpatchs = []
    realpatchs = []

    _,h, w,_ = K.int_shape(img)[:]
    size = int(h // 4)
    for i in range(N):
        rand_coe_h = random.random() * (y - x) + x
        rand_coe_w = random.random() * (y - x) + x

        # get width and height of the patch
        rand_h = int(h * rand_coe_h)
        rand_w = int(w * rand_coe_w)

        # the random - generated coordinates are limited in
        # h -> [0, coor_h]
        # w -> [0, coor_w]
        coor_h = h - rand_h
        coor_w = w - rand_w

        # get x and y starting point of the patch
        coor_x = int(random.random() * coor_h)
        coor_y = int(random.random() * coor_w)

        # only create patches for the high layer
        img_patch = img[:,coor_x:coor_x + rand_h, coor_y:coor_y + rand_w,:]
        # resize the patch to [size, size]
        resize_img = np.resize(img_patch, (BATCH_SIZE,size, size,OUT_CH))
        imgpatchs.append(resize_img)

        # Create patches for the label
        label_patch = label[:,coor_x:coor_x + rand_h, coor_y:coor_y + rand_w,:]
        # resize the patch to [size, size]
        resize_label = np.resize(label_patch, (BATCH_SIZE,size, size,OUT_CH*3))
        labelpatchs.append(resize_label)

        # Create patches for the label
        real_patch = real[:, coor_x:coor_x + rand_h, coor_y:coor_y + rand_w, :]
        # resize the patch to [size, size]
        resize_label = np.resize(real_patch, (BATCH_SIZE,size, size,OUT_CH))
        realpatchs.append(resize_label)

    labelpatchs_ = np.resize(labelpatchs, (N,BATCH_SIZE,size, size,OUT_CH*3))
    imgpatchs_ = np.resize(imgpatchs, (N,BATCH_SIZE,size, size,OUT_CH))
    realpatchs_ = np.resize(realpatchs, (N,BATCH_SIZE,size, size,OUT_CH))
    return [labelpatchs_, imgpatchs_, realpatchs_]

def slice(x,i):
    """ Define a tensor slice function
    """
    return x[i:i+1, :, :, :]

def generator_containing_discriminator(generator, discriminator):
    inputs = Input((img_cols, img_rows, IN_CH * 3))
    x_generator = generator(inputs)

    merged_global = Concatenate(axis=3)([inputs, x_generator])
    real_local, fake_local = KL.Lambda(lambda x: crop_random(*x), name='crop_local')([inputs, x_generator])
    discriminator.trainable = False

    x1 = KL.Lambda(slice, arguments={'i': 0})(real_local)
    squeezed1 = KL.Lambda(lambda x: K.squeeze(x1, 0))(x1)
    x2 = KL.Lambda(slice, arguments={'i': 0})(fake_local)
    squeezed2 = KL.Lambda(lambda x: K.squeeze(x2, 0))(x2)
    merged_local = Concatenate(axis=3)([squeezed1, squeezed2])
    x_discriminator0 = discriminator([merged_global, merged_local])

    x1 = KL.Lambda(slice, arguments={'i': 1})(real_local)
    squeezed1 =  KL.Lambda(lambda x: K.squeeze(x1, 0))(x1)
    x2 =  KL.Lambda(slice, arguments={'i': 1})(fake_local)
    squeezed2 =  KL.Lambda(lambda x: K.squeeze(x2, 0))(x2)
    merged_local = Concatenate(axis=3)([squeezed1, squeezed2])
    x_discriminator1 = discriminator([merged_global, merged_local])

    x1 =  KL.Lambda(slice, arguments={'i': 2})(real_local)
    squeezed1 =  KL.Lambda(lambda x: K.squeeze(x1, 0))(x1)
    x2 =  KL.Lambda(slice, arguments={'i': 2})(fake_local)
    squeezed2 =  KL.Lambda(lambda x: K.squeeze(x2, 0))(x2)
    merged_local = Concatenate(axis=3)([squeezed1, squeezed2])
    x_discriminator2 = discriminator([merged_global, merged_local])

    x1 = KL.Lambda(slice, arguments={'i': 3})(real_local)
    squeezed1 = KL.Lambda(lambda x: K.squeeze(x1, 0))(x1)
    x2 =  KL.Lambda(slice, arguments={'i': 3})(fake_local)
    squeezed2 =  KL.Lambda(lambda x: K.squeeze(x2, 0))(x2)
    merged_local = Concatenate(axis=3)([squeezed1, squeezed2])
    x_discriminator3 = discriminator([merged_global, merged_local])

    x1 = KL.Lambda(slice, arguments={'i': 4})(real_local)
    squeezed1 = KL.Lambda(lambda x: K.squeeze(x1, 0))(x1)
    x2 = KL.Lambda(slice, arguments={'i': 4})(fake_local)
    squeezed2 = KL.Lambda(lambda x: K.squeeze(x2, 0))(x2)
    merged_local = Concatenate(axis=3)([squeezed1, squeezed2])
    x_discriminator4 = discriminator([merged_global, merged_local])

    model = Model(input=inputs, output=[x_generator, x_discriminator0, x_discriminator1, x_discriminator2, x_discriminator3,x_discriminator4])
    return model


def discriminator_loss(y_true, y_pred):
    global BATCH_SIZE
    return K.mean(K.square(K.flatten(y_pred) - K.concatenate([K.ones_like(K.flatten(y_pred[:BATCH_SIZE, :])),
                                                              K.zeros_like(K.flatten(y_pred[:BATCH_SIZE, :]))])),
                  axis=-1)


def discriminator_on_generator_loss(y_true, y_pred):
    global BATCH_SIZE
    return K.mean(K.square(K.flatten(y_pred) - K.ones_like(K.flatten(y_pred))), axis=-1)


def generator_l1_loss(y_true, y_pred):
    global BATCH_SIZE
    return K.mean(K.abs(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)


def named_logs(model, logs):
    result = {}
    logs_list = []
    logs_list.append(logs)
    model_list = []
    model_list.append(model)
    zipped = zip(model_list, logs_list)
    zip_list = list(zipped)
    for l in zip_list:
        result[l[0]] = l[1]
    return result

n_critic = 5
clip_value = 0.01
def train(BATCH_SIZE, epochs=1000, attention=False):

    sess = K.get_session()
    tfrecord_list = glob(os.path.join(train_path, '**', '*.tfrecords'))
    shuffle(tfrecord_list)
    filename_queue = tf.train.string_input_producer(tfrecord_list, num_epochs=epochs + 1, shuffle=True)
    in_LDRs, _, _, ref_HDR, _, _ = load_data(filename_queue, BATCH_SIZE, img_rows, img_rows, IN_CH,
                                             num_shots)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    discriminator = discriminator_all()
    generator = generator_model(attention=attention)

    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    optimizer = Adam(0.0002, 0.5)
    if os.path.isfile('generator'+'_wgan_local.h5'):
        generator.load_weights('generator' +'_wgan_local.h5')
    generator.compile(loss='mse', optimizer=optimizer)
    discriminator_on_generator.compile(loss=[generator_l1_loss, discriminator_on_generator_loss, discriminator_on_generator_loss, discriminator_on_generator_loss, discriminator_on_generator_loss, discriminator_on_generator_loss],
                                       optimizer=optimizer)
    discriminator.trainable = True
    if os.path.isfile('discriminator' +'_wgan_local.h5'):
        discriminator.load_weights('discriminator' +'_wgan_local.h5')
    discriminator.compile(loss=discriminator_loss, optimizer=optimizer)
    epoch_star_time = time.time()

    tensorboard_2.set_model(discriminator)
    tensorboard_3.set_model(discriminator_on_generator)
    writer = tf.summary.FileWriter(result_path+'logs/')

    for epoch in range(epochs):
        print("Epoch : %d/%d" % (epoch, epochs))

        for index in range(steps):
            global step_count
            batch_star_time = time.time()
            X_train, Y_train = sess.run([in_LDRs, ref_HDR])
            image_batch = tonemap_np(Y_train)  # [index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            for _ in range(n_critic):
                generated_images = generator.predict(X_train)

                real_pairs = np.concatenate((X_train, image_batch), axis=3)
                fake_pairs = np.concatenate((X_train, generated_images), axis=3)
                X = np.concatenate((real_pairs, fake_pairs))
                y = np.zeros((2 * BATCH_SIZE, 64,64,1))  # [1] * BATCH_SIZE + [0] * BATCH_SIZE
                y1 = np.zeros((2 * BATCH_SIZE, 64, 64, 1))
                condition, fake, real = crop_random0(generated_images, X_train, image_batch)

                for n in range(5):
                    print(K.int_shape(condition[n]))
                    real_pairs_local = np.concatenate((condition[n], real[n]), axis=3)
                    fake_pairs_local = np.concatenate((condition[n], fake[n]), axis=3)
                    X_local = np.concatenate((real_pairs_local, fake_pairs_local))

                    d_loss = discriminator.train_on_batch([X, X_local], y)

                    tensorboard_2.on_epoch_end(step_count, named_logs('d_loss', d_loss))
                for l in discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                    l.set_weights(weights)

            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(X_train, [image_batch,np.ones((BATCH_SIZE,1)),np.ones((BATCH_SIZE,1)),np.ones((BATCH_SIZE,1)),np.ones((BATCH_SIZE,1)),np.ones((BATCH_SIZE,1))])
            tensorboard_3.on_epoch_end(step_count, named_logs('g_loss', g_loss[1]))
            discriminator.trainable = True

            if step_count % 100 == 0:
                ref = tf.summary.image("G_ref/{}".format(step_count), Y_train, max_outputs=12)
                input_1 = tf.summary.image("input_1/{}".format(step_count), X_train[:, :, :, 0:3], max_outputs=12)
                input_2 = tf.summary.image("input_2/{}".format(step_count), X_train[:, :, :, 3:6], max_outputs=12)
                input_3 = tf.summary.image("input_3/{}".format(step_count), X_train[:, :, :, 6:9], max_outputs=12)
                gen = tf.summary.image("gen/{}".format(step_count), generated_images, max_outputs=12)
                merge_sum = tf.summary.merge([ref, input_1, input_2, input_3, gen])
                sum = sess.run(merge_sum)
                writer.add_summary(sum, step_count)
                writer.flush()
                his_log_dis.log_weights(discriminator, step_count)
                his_log_gen.log_weights(generator, step_count)

            print("epoch:%d, batch:%d/%d, d_loss:%f, g_loss:%f, time:%f" % (
            epoch, index, steps, d_loss, g_loss[1], time.time() - batch_star_time))
            step_count += 1

        generator.save_weights('generator'+'_wgan_local' + '.h5', True)
        discriminator.save_weights('discriminator'+'_wgan_local' + '.h5', True)

    print("epoch_total_time : %f" % (time.time() - epoch_star_time))
    coord.request_stop()
    coord.join(threads)


def test(BATCH_SIZE, nice=False, attention=False):
    suffix = ''
    if attention:
        suffix = '_attention'
    generator = generator_model(test=True)
    optimizer = Adam(0.0002, 0.5)
    generator.compile(loss='mse', optimizer=optimizer)
    generator.load_weights('generator' + suffix+'_wgan_local' + '.h5')

    scene_dirs = sorted(os.listdir(test_path))
    nScenes = len(scene_dirs)
    num_batch = int(np.ceil(nScenes / BATCH_SIZE))

    psnr_f = open(os.path.join(result_path, 'psnr.txt'), 'w')
    psnr = []
    for idx in range(0, num_batch):
        print('batch no. %d:' % (idx + 1))
        psnr_f.write('batch no. %d:\n' % (idx + 1))

        l = idx * BATCH_SIZE
        u = min((idx + 1) * BATCH_SIZE, nScenes)
        batchSz = u - l
        batch_scene_dirs = scene_dirs[l:u]
        batch_in_LDRs = []
        batch_in_HDRs = []
        batch_ref_HDR = []
        for i, scene_dir in enumerate(batch_scene_dirs):
            _LDRs, _HDRs, _HDR = get_input(os.path.join(test_path, scene_dir),
                                           [test_h, test_w])

            batch_in_LDRs = batch_in_LDRs + [_LDRs]
            batch_in_HDRs = batch_in_HDRs + [_HDRs]
            batch_ref_HDR = batch_ref_HDR + [_HDR]
        batch_in_LDRs = np.array(batch_in_LDRs, dtype=np.float32)
        batch_in_HDRs = np.array(batch_in_HDRs, dtype=np.float32)
        batch_ref_HDR = np.array(batch_ref_HDR, dtype=np.float32)

        # deal with last batch
        if batchSz < BATCH_SIZE:
            print(batchSz)
            padSz = ((0, int(BATCH_SIZE - batchSz)), (0, 0), (0, 0), (0, 0))
            batch_in_LDRs = np.pad(batch_in_LDRs, padSz, 'wrap').astype(np.float32)
            batch_in_HDRs = np.pad(batch_in_HDRs, padSz, 'wrap').astype(np.float32)

        st = time.time()
        generated_images = generator.predict(batch_in_LDRs)
        print("time: %.4f" % (time.time() - st))

        curr_psnr = [compute_psnr(generated_images[i], tonemap_np(batch_ref_HDR[i])) for i in range(batchSz)]
        print("PSNR: %.4f\n" % np.mean(curr_psnr))
        psnr_f.write("PSNR: %.4f\n\n" % np.mean(curr_psnr))
        psnr += curr_psnr

        save_results([batch_ref_HDR],
                     os.path.join(result_path, 'test_{:03d}_{:03d}_ref_HDR.hdr'.format(l, u)))
        save_results([generated_images[:batchSz]],
                     os.path.join(result_path, 'test_{:03d}_{:03d}_tonemapped.png'.format(l, u)))
        save_results([batch_in_LDRs],
                     os.path.join(result_path, 'test_{:03d}_{:03d}_LDRs.png'.format(l, u)))

    avg_psnr = np.mean(psnr)
    print("Average PSNR: %.4f" % avg_psnr)
    psnr_f.write("Average PSNR: %.4f" % avg_psnr)
    psnr_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--data', help='A directory of 512x256 images')
    parser.add_argument('--mode', choices=['AtoB', 'BtoA'], default='AtoB')
    global args
    args = parser.parse_args()
    # train(BATCH_SIZE=BATCH_SIZE,epochs=11, attention=False)
    test(BATCH_SIZE=batch_size, attention=False)
