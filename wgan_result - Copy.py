import numpy as np
import pickle, sys, keras,argparse,cv2
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge, LeakyReLU, Lambda
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
from keras.layers import GaussianNoise, Conv2D
from keras.initializers import RandomNormal
IN_CH = 3
OUT_CH = 3
NF = 64  # number of filter
BATCH_SIZE = 20
num_shots = 3
result_path = 'wgan_results/'
sample_path = 'samples/'
log_path = 'log/'
step_count = 0

##test config
test_path = '/home/idst/workspace/ziyi_dataset/dataset/align/Test/'
test_h = 768  # 1024
test_w = 1280  # 1536
batch_size = 1
##train config
train_path = '/home/idst/workspace/ziyi_dataset/dataset/tf_records/'
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


def get_input(scene_dir, image_size):
    c = 3
    in_LDR_paths = sorted(glob(os.path.join(scene_dir, '2*.tif')))
    ns = len(in_LDR_paths)

    in_exps_path = os.path.join(scene_dir, 'exposure.txt')
    in_exps = np.array(open(in_exps_path).read().split('\n')[:ns]).astype(np.float32)
    in_exps -= in_exps.min()
    in_LDRs = np.zeros(image_size + [c * ns], dtype=np.float32)
    in_HDRs = np.zeros(image_size + [c * ns], dtype=np.float32)

    for i, image_path in enumerate(in_LDR_paths):
        img = get_image(image_path, image_size=image_size, is_crop=True)
        in_LDRs[:, :, c * i:c * (i + 1)] = img
        in_HDRs[:, :, c * i:c * (i + 1)] = LDR2HDR(img, 2. ** in_exps[i])

    ref_HDR = get_image(os.path.join(scene_dir, 'HDRImg.hdr'), image_size=image_size, is_crop=True)
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

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)
########## Additional Helper Functions ##########

def generator_model(test=False):
    global BATCH_SIZE
    # imgs: input: 256x256xch
    # U-Net structure, must change to relu
    global img_rows, img_cols
    if test:
        img_rows = test_h
        img_cols = test_w
    inputs1 = Input((img_rows, img_cols, IN_CH * 3))
    # inputs2 = Input((img_rows, img_cols, IN_CH))
    # inputs = Concatenate(axis=3)([inputs1, inputs2])
    f_size = 5
    e1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(inputs1)
    e1 = Convolution2D(64, f_size, f_size, subsample=(2, 2), init='uniform', border_mode='same')(e1)
    e1 = LeakyReLU(alpha=0.02)(e1)
    e1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e1)
    e2 = Convolution2D(128, f_size, f_size, subsample=(2, 2), init='uniform', border_mode='same')(e1)
    e2 = LeakyReLU(alpha=0.02)(e2)
    e2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e2)

    e3 = Convolution2D(256, f_size, f_size, subsample=(2, 2), init='uniform', border_mode='same')(e2)
    e3 = LeakyReLU(alpha=0.02)(e3)
    e3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e3)
    e4 = Convolution2D(512, f_size, f_size, subsample=(2, 2), init='uniform', border_mode='same')(e3)
    e4 = LeakyReLU(alpha=0.02)(e4)
    e4 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e4)

    e5 = Convolution2D(512, f_size, f_size, subsample=(2, 2), init='uniform', border_mode='same')(e4)
    e5 = LeakyReLU(alpha=0.02)(e5)
    e5 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e5)
    e6 = Convolution2D(512, f_size, f_size, subsample=(2, 2), init='uniform', border_mode='same')(e5)
    e6 = LeakyReLU(alpha=0.02)(e6)
    e6 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e6)

    e7 = Convolution2D(512, f_size, f_size, subsample=(2, 2), init='uniform', border_mode='same')(e6)
    e7 = LeakyReLU(alpha=0.02)(e7)
    e7 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e7)
    e8 = Convolution2D(512, f_size, f_size, subsample=(2, 2), init='uniform', border_mode='same')(e7)
    e8 = LeakyReLU(alpha=0.02)(e8)
    e8 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e8)

    d1 = UpSampling2D(interpolation='nearest')(e8)
    d1 = Convolution2D(512, f_size, f_size, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d1)
    d1 = Concatenate(axis=3)([d1, e7])
    d1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d1)

    d2 = UpSampling2D(interpolation='nearest')(d1)
    d2 = Convolution2D(512, f_size, f_size, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d2)
    d2 = Concatenate(axis=3)([d2, e6])
    d2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d2)

    d3 = Dropout(0.2)(d2)
    d3 = UpSampling2D(interpolation='nearest')(d3)
    d3 = Convolution2D(512, f_size, f_size, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d3)
    d3 = Concatenate(axis=3)([d3, e5])
    d3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d3)

    d4 = Dropout(0.2)(d3)
    d4 = UpSampling2D(interpolation='nearest')(d4)
    d4 = Convolution2D(512, f_size, f_size, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d4)
    d4 = Concatenate(axis=3)([d4, e4])
    d4 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d4)

    d5 = Dropout(0.2)(d4)
    d5 = UpSampling2D(interpolation='nearest')(d5)
    d5 = Convolution2D(256, f_size, f_size, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d5)
    d5 = Concatenate(axis=3)([d5, e3])
    d5 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d5)

    d6 = Dropout(0.2)(d5)
    d6 = UpSampling2D(interpolation='nearest')(d6)
    d6 = Convolution2D(128, f_size, f_size, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d6)
    d6 = Concatenate(axis=3)([d6, e2])
    d6 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d6)

    d7 = Dropout(0.2)(d6)
    d7 = UpSampling2D(interpolation='nearest')(d7)
    d7 = Convolution2D(64, f_size, f_size, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d7)
    d7 = Concatenate(axis=3)([d7, e1])

    d7 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d7)
    d8 = UpSampling2D(interpolation='nearest')(d7)
    d8 = Convolution2D(3, f_size, f_size, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d8)
    d8 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d8)
    d9 = Activation('tanh')(d8)

    model = Model(input=inputs1, output=d9)
    return model


def discriminator_model():
    """ return a (b, 1) logits"""
    f_size = 5
    model = Sequential()
    model.add(Convolution2D(64, f_size, f_size, border_mode='same', input_shape=(img_rows, img_cols, IN_CH * 4)))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, f_size, f_size, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(512, f_size, f_size, border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Convolution2D(1, f_size, f_size, border_mode='same'))
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
    inputs1 = Input((img_cols, img_rows, IN_CH * 3))

    x_generator = generator(inputs1)

    merged = Concatenate(axis=3)([inputs1, x_generator])
    discriminator.trainable = False
    x_discriminator = discriminator(merged)

    model = Model(input=inputs1, output=[x_generator, x_discriminator])

    return model


def discriminator_loss(y_true, y_pred):
    return K.mean(K.square(K.flatten(y_pred) - K.concatenate([K.ones_like(K.flatten(y_pred[:BATCH_SIZE, :])),
                                                              K.zeros_like(K.flatten(y_pred[:BATCH_SIZE, :]))])),
                  axis=-1)
def generator_l1_loss(y_true, y_pred):
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
def train(BATCH_SIZE, epochs=1000):
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

    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    # if os.path.isfile('generator_wgan.h5'):
    #     generator.load_weights('generator_wgan.h5')
    # if os.path.isfile('discriminator_wgan.h5'):
    #     discriminator.load_weights('discriminator_wgan.h5')

    generator.compile(loss='mae', optimizer=RMSprop(lr=0.00005))
    discriminator_on_generator.compile(loss=[generator_l1_loss, 'mse'], optimizer=RMSprop(lr=0.00005))
    discriminator.trainable = True
    discriminator.compile(loss=discriminator_loss, optimizer=RMSprop(lr=0.00005))

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
                # noise0 = np.random.uniform(-1.0, 1.0, [BATCH_SIZE, 256, 256, 3])
                generated_images = generator.predict(X_train)

                real_pairs = np.concatenate((X_train, image_batch), axis=3)
                fake_pairs = np.concatenate((X_train, generated_images), axis=3)
                X = np.concatenate((real_pairs, fake_pairs))
                y = np.zeros((2 * BATCH_SIZE, 64, 64, 1))
                d = discriminator.train_on_batch(X, y)
                tensorboard_2.on_epoch_end(step_count, named_logs('d', d))

                # real_y = np.ones((BATCH_SIZE, 64, 64, 1))
                # fake_y = np.ones((BATCH_SIZE, 64, 64, 1)) * -1
                # d_f = discriminator.train_on_batch(fake_pairs, fake_y)
                # d_r = discriminator.train_on_batch(real_pairs, real_y)
                # tensorboard_2.on_epoch_end(step_count, named_logs('d_f', d_f))
                # tensorboard_2.on_epoch_end(step_count, named_logs('d_r', d_r))

                for l in discriminator.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                    l.set_weights(weights)
            #################### wgan ####################
            discriminator.trainable = False
            fake_y = np.ones((BATCH_SIZE, 64, 64, 1))
            # noise = np.random.uniform(-1.0, 1.0, [BATCH_SIZE, 256,256,3])
            g_loss = discriminator_on_generator.train_on_batch(X_train, [image_batch, fake_y])
            tensorboard_3.on_epoch_end(step_count, named_logs('mas', g_loss[1]))
            tensorboard_3.on_epoch_end(step_count, named_logs('g_r', g_loss[2]))

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

            print("epoch:%d, batch:%d/%d, d:%f, g_loss:%f, time:%f" % (
                epoch, index, steps, d,  g_loss[1], time.time() - batch_star_time))
            step_count += 1

        # generator.save_weights('generator_wgan.h5', True)
        # discriminator.save_weights('discriminator_wgan.h5', True)

    print("epoch_total_time : %f" % (time.time() - epoch_star_time))
    coord.request_stop()
    coord.join(threads)


import sys
import numpy
from scipy import signal
from scipy import ndimage

import gauss

def ssim_value(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = gauss.fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))

def test(BATCH_SIZE):
    generator = generator_model(test=True)
    optimizer = Adam(0.0002, 0.5)
    generator.compile(loss='mse', optimizer=optimizer)
    generator.load_weights('./weight/generator_wgan.h5')

    scene_dirs = sorted(os.listdir(args.test_path))
    nScenes = len(scene_dirs)
    num_batch = int(np.ceil(nScenes / BATCH_SIZE))

    psnr_f = open(os.path.join(result_path, 'psnr.txt'), 'w')
    psnr = []
    ###########
    ssim = []
    #########
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
        # batch_in_LDRs[:,:,:,3:6] = batch_in_LDRs[:,:,:,0:3]
        batch_in_HDRs = np.array(batch_in_HDRs, dtype=np.float32)
        batch_ref_HDR = np.array(batch_ref_HDR, dtype=np.float32)

        # deal with last batch
        if batchSz < BATCH_SIZE:
            print(batchSz)
            padSz = ((0, int(BATCH_SIZE - batchSz)), (0, 0), (0, 0), (0, 0))
            batch_in_LDRs = np.pad(batch_in_LDRs, padSz, 'wrap').astype(np.float32)
            batch_in_HDRs = np.pad(batch_in_HDRs, padSz, 'wrap').astype(np.float32)

        st = time.time()
        _,w,h,_ = batch_in_LDRs.shape
        # noise0 = np.random.uniform(-1.0, 1.0, [BATCH_SIZE,w,h,3])
        generated_images = generator.predict(batch_in_LDRs)
        print("time: %.4f" % (time.time() - st))

        curr_psnr = [compute_psnr(generated_images[i], tonemap_np(batch_ref_HDR[i])) for i in range(batchSz)]
        print("PSNR: %.4f\n" % np.mean(curr_psnr))
        psnr_f.write("PSNR: %.4f\n\n" % np.mean(curr_psnr))
        psnr += curr_psnr
        ########
        # curr_ssim = [ssim_value(cv2.cvtColor(tonemap_np(batch_ref_HDR[i]), cv2.COLOR_BGR2GRAY), cv2.cvtColor(tonemap_np(generated_images[i]),cv2.COLOR_BGR2GRAY)) for i in range(batchSz)]
        # print("ssim: %.4f\n" % np.mean(curr_ssim))
        # psnr_f.write("ssim: %.4f\n\n" % np.mean(curr_ssim))
        # ssim += curr_ssim
        ########
        # save_results([batch_ref_HDR],
        #              os.path.join(result_path, 'test_{:03d}_{:03d}_ref_HDR.hdr'.format(l, u)))
        save_results([generated_images[:batchSz]],
                     os.path.join(result_path, 'test_{:03d}_{:03d}_tonemapped.png'.format(l, u)))
        save_results([batch_in_LDRs],
                     os.path.join(result_path, 'test_{:03d}_{:03d}_LDRs.png'.format(l, u)))

    avg_psnr = np.mean(psnr)
    print("Average PSNR: %.4f" % avg_psnr)
    psnr_f.write("Average PSNR: %.4f" % avg_psnr)
    #########
    # avg_ssim = np.mean(ssim)
    # print("Average SSIM: %.4f" % avg_ssim)
    # psnr_f.write("Average SSIM: %.4f" % avg_ssim)
    #########
    psnr_f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train')
    parser.add_argument('--test_path', help='test_path')
    parser.add_argument('--train_path', help='train_path')
    global args
    args = parser.parse_args()
    if (args.train):
        train(BATCH_SIZE=BATCH_SIZE,epochs=20)
    else:
        test(BATCH_SIZE=batch_size)
