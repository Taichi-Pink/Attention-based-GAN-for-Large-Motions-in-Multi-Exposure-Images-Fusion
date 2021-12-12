import numpy as np
import pickle
import sys, keras
import argparse
import cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge, LeakyReLU
from keras.layers import Reshape, Concatenate
from keras.layers.core import Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers.core import Flatten
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, Adagrad, RMSprop
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
import tensorflow.contrib as tf_contrib
weight_init = tf_contrib.layers.xavier_initializer()
weight_regularizer = None
weight_regularizer_fully = None
IN_CH = 3
OUT_CH = 3
NF = 64  # number of filter
BATCH_SIZE = 20
num_shots = 3
result_path = 'wgan_attention/'
sample_path = 'samples/'
log_path = 'log/'
step_count = 0

##test config
test_path = '/home/ziyi.liu1/dataset_new/ICCP19_train_set/Testing_set/'#'/home/ziyi.liu1/dataset/test/'
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
    log_dir=result_path + 'log1/',
    batch_size=BATCH_SIZE,
    write_graph=True,
    write_grads=True
)
tensorboard_3 = keras.callbacks.TensorBoard(
    log_dir=result_path + 'log2/',
    batch_size=BATCH_SIZE,
    write_graph=True,
    write_grads=True
)
his_log_gen = tfboard_loggers.TFBoardKerasModelWeightsLogger(result_path + "gen/")
his_log_dis = tfboard_loggers.TFBoardKerasModelWeightsLogger(result_path + "dis/")


def tonemap_np(images):  # input/output -1~1
    return np.log(1 + MU * (images + 1) / 2.) / np.log(1 + MU) * 2. - 1

# new data
def get_input0(scene_dir, image_size):
    c = 3
    in_LDR_paths = sorted(glob(os.path.join(scene_dir, 'ghosted', '*.tif')))  # *aligned.tif
    ns = len(in_LDR_paths)
    # print(ns)
    # print(in_LDR_paths)

    in_exps_path = os.path.join(scene_dir, 'ExpoTimes.txt')  # input_exp.txt
    in_exps = np.array(open(in_exps_path).read().split('\n')[:ns]).astype(np.float32)

    in_exps -= in_exps.min()
    in_LDRs = np.zeros(image_size + [c * ns], dtype=np.float32)
    in_HDRs = np.zeros(image_size + [c * ns], dtype=np.float32)

    for i, image_path in enumerate(in_LDR_paths):
        img = get_image(image_path, image_size=image_size, is_crop=True)
        in_LDRs[:, :, c * i:c * (i + 1)] = img
        in_HDRs[:, :, c * i:c * (i + 1)] = LDR2HDR(img, 2. ** in_exps[i])

    ref_HDR = get_image(os.path.join(scene_dir, 'GT_HDR.hdr'), image_size=image_size, is_crop=True)  # ref_hdr_aligned
    return in_LDRs, in_HDRs, ref_HDR
    # return in_LDRs, in_HDRs

def get_input(scene_dir, image_size):
    c = 3
    in_LDR_paths = sorted(glob(os.path.join(scene_dir, '2*.tif')))  # *aligned.tif
    ns = len(in_LDR_paths)
    # print(ns)
    # print(in_LDR_paths)

    in_exps_path = os.path.join(scene_dir, 'exposure.txt')
    in_exps = np.array(open(in_exps_path).read().split('\n')[:ns]).astype(np.float32)

    in_exps -= in_exps.min()
    in_LDRs = np.zeros(image_size + [c * ns], dtype=np.float32)
    in_HDRs = np.zeros(image_size + [c * ns], dtype=np.float32)

    for i, image_path in enumerate(in_LDR_paths):
        img = get_image(image_path, image_size=image_size, is_crop=True)
        in_LDRs[:, :, c * i:c * (i + 1)] = img
        in_HDRs[:, :, c * i:c * (i + 1)] = LDR2HDR(img, 2. ** in_exps[i])

    ref_HDR = get_image(os.path.join(scene_dir, 'HDRImg.hdr'), image_size=image_size, is_crop=True)  # ref_hdr_aligned
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
def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True,sn=False, scope='conv_0'):
    #
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride== 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x

def max_pooling(x, scope='max_pooling') :
    with tf.variable_scope(scope):
        y = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        return y

def hw_flatten(x) :
    return tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2], x.shape[-1]])


def google_attention(x,scope='attention'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        channels = K.int_shape(x)[-1]

        batch_size, height, width, num_channels = x.get_shape().as_list()
        # f = KL.Lambda(lambda x: conv(*x), arguments={'channels':channels // 8, 'kernel':1,'stride':1, 'sn':True})([x])# [bs, h, w, c']
        # f = KL.Lambda(lambda x: max_pooling(*x))([f])
        f = conv(x, channels // 8, kernel=1, stride=1, sn=True,scope='f_conv')  # [bs, h, w, c']
        print(K.int_shape(f)[:])
        f = max_pooling(f, scope='f_pooling')
        g = conv(x, channels // 8, kernel=1, stride=1, sn=True, scope='g_conv')  # [bs, h, w, c']

        h = conv(x, channels // 2, kernel=1, stride=1, sn=True, scope='h_conv')  # [bs, h, w, c]
        h = max_pooling(h)

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=[-1, height, width, num_channels // 2])  # [bs, h, w, C]
        o = conv(o, channels, kernel=1, stride=1, sn=True, scope='o_conv')
        x0 = tf.add(tf.multiply(gamma, o), x)

        return x0
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
    e2 = KL.Lambda(lambda x: google_attention(x), name='crop_local')(e2)

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


def generator_containing_discriminator(generator, discriminator):
    inputs = Input((img_cols, img_rows, IN_CH * 3))
    x_generator = generator(inputs)

    merged = Concatenate(axis=3)([inputs, x_generator])
    discriminator.trainable = False
    x_discriminator = discriminator(merged)

    model = Model(input=inputs, output=[x_generator, x_discriminator])

    return model


def discriminator_loss(y_true, y_pred):
    global BATCH_SIZE
    return K.mean(K.square(K.flatten(y_pred) - K.concatenate([K.ones_like(K.flatten(y_pred[:BATCH_SIZE, :, :, :])),
                                                              K.zeros_like(K.flatten(y_pred[:BATCH_SIZE, :, :, :]))])),
                  axis=-1)


def discriminator_on_generator_loss(y_true, y_pred):
    global BATCH_SIZE
    return K.mean(K.square(K.flatten(y_true)- K.ones_like(K.flatten(y_pred))), axis=-1)


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
    # suffix = ''
    # if attention:
    #     suffix = '_attention'

    sess = K.get_session()
    tfrecord_list = glob(os.path.join(args.train_path, '*.tfrecords'))
    assert (tfrecord_list)
    shuffle(tfrecord_list)
    filename_queue = tf.train.string_input_producer(tfrecord_list, num_epochs=epochs + 1, shuffle=True)
    in_LDRs, _, _, ref_HDR, _, _ = load_data(filename_queue, BATCH_SIZE, img_rows, img_rows, IN_CH,
                                             num_shots)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    discriminator = discriminator_model(attention=attention)
    generator = generator_model(attention=attention)
    generator.summary()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)
    optimizer = RMSprop(lr=0.00005)
    # if os.path.isfile('generator_wgan_attention.h5'):
    #     generator.load_weights('generator_wgan_attention' + suffix + '.h5')
    generator.compile(loss='mse', optimizer=optimizer)
    discriminator_on_generator.compile(loss=[generator_l1_loss, discriminator_on_generator_loss], optimizer=optimizer)
    discriminator.trainable = True
    # if os.path.isfile('discriminator_wgan_attention' + suffix + '.h5'):
    #     discriminator.load_weights('discriminator_wgan_attention' + suffix + '.h5')
    discriminator.compile(loss=discriminator_loss, optimizer=optimizer)
    epoch_star_time = time.time()

    tensorboard_2.set_model(discriminator)
    tensorboard_3.set_model(discriminator_on_generator)
    # writer = tf.summary.FileWriter(result_path + 'logs/')

    for epoch in range(epochs):
        print("Epoch : %d/%d" % (epoch, epochs))

        for index in range(steps):
            global step_count
            batch_star_time = time.time()
            X_train, Y_train = sess.run([in_LDRs, ref_HDR])
            image_batch = tonemap_np(Y_train)
            #################### wgan ####################
            for _ in range(n_critic):
                generated_images = generator.predict(X_train)

                real_pairs = np.concatenate((X_train, image_batch), axis=3)
                fake_pairs = np.concatenate((X_train, generated_images), axis=3)
                X = np.concatenate((real_pairs, fake_pairs))
                y = np.zeros((2 * BATCH_SIZE, 64, 64, 1))
                d_loss = discriminator.train_on_batch(X, y)
                tensorboard_2.on_epoch_end(step_count, named_logs('d_loss', d_loss))

                for l in discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                    l.set_weights(weights)
            #################### wgan ####################
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(X_train, [image_batch, np.ones((BATCH_SIZE, 64, 64, 1))])
            tensorboard_3.on_epoch_end(step_count, named_logs('g_loss', g_loss[1]))
            discriminator.trainable = True

            # if step_count % 100 == 0:
            #     ref = tf.summary.image("G_ref/{}".format(step_count), Y_train, max_outputs=12)
            #     input_1 = tf.summary.image("input_1/{}".format(step_count), X_train[:, :, :, 0:3], max_outputs=12)
            #     input_2 = tf.summary.image("input_2/{}".format(step_count), X_train[:, :, :, 3:6], max_outputs=12)
            #     input_3 = tf.summary.image("input_3/{}".format(step_count), X_train[:, :, :, 6:9], max_outputs=12)
            #     gen = tf.summary.image("gen/{}".format(step_count), generated_images, max_outputs=12)
            #     merge_sum = tf.summary.merge([ref, input_1, input_2, input_3, gen])
            #     sum = sess.run(merge_sum)
            #     writer.add_summary(sum, step_count)
            #     writer.flush()
            #     his_log_dis.log_weights(discriminator, step_count)
            #     his_log_gen.log_weights(generator, step_count)

            print("epoch:%d, batch:%d/%d, d_loss:%f, g_loss:%f, time:%f" % (
                epoch, index, steps, d_loss, g_loss[1], time.time() - batch_star_time))
            step_count += 1

        # generator.save_weights('generator_wgan_attention' + suffix + '.h5', True)
        # discriminator.save_weights('discriminator_wgan_attention' + suffix + '.h5', True)

    print("epoch_total_time : %f" % (time.time() - epoch_star_time))
    coord.request_stop()
    coord.join(threads)


def test(BATCH_SIZE, attention=False):
    suffix = ''
    if attention:
        suffix = '_attention'
    generator = generator_model(test=True)
    optimizer = Adam(0.0002, 0.5)
    generator.compile(loss='mse', optimizer=optimizer)
    generator.load_weights('./weight/generator_wgan_attention.h5')

    scene_dirs = sorted(os.listdir(args.test_path))
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
            _LDRs, _HDRs, _HDR = get_input(os.path.join(args.test_path, scene_dir),
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

        # save_results([batch_ref_HDR],
        #              os.path.join(result_path, 'test_{:03d}_{:03d}_ref_HDR.hdr'.format(l, u)))
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
    parser.add_argument('--train', help='train')
    parser.add_argument('--test_path', help='test_path')
    parser.add_argument('--train_path', help='train_path')
    global args
    args = parser.parse_args()
    if (args.train):
        train(BATCH_SIZE=BATCH_SIZE,epochs=50, attention=False)
    else:
        test(BATCH_SIZE=batch_size, attention=False)
