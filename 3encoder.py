import numpy as np
import pickle,sys, keras, argparse, cv2, math, imageio, time, os
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge, LeakyReLU,Add, Reshape, Concatenate, Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Deconvolution2D
from keras.optimizers import SGD, Adagrad
from PIL import Image
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
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
NF = 64  # number of filter
BATCH_SIZE = 20
num_shots = 3
result_path = '3encoder_result/'
sample_path = 'samples/'
log_path = 'log/'
step_count = 0

##test config
test_path = '/home/ziyi.liu1/dataset/test/'
batch_size = 1
##train config
train_path = '/home/ziyi.liu1/dataset/tf_records/'
img_rows = 256
img_cols = 256
total_images = 20202
steps = 955
MU = 5000.

tensorboard_2 = keras.callbacks.TensorBoard(
    log_dir='3encoder_result/log1/',
    batch_size=BATCH_SIZE,
    write_graph=True,
    write_grads=True
)
tensorboard_3 = keras.callbacks.TensorBoard(
    log_dir='3encoder_result/log2/',
    batch_size=BATCH_SIZE,
    write_graph=True,
    write_grads=True
)

his_log_gen = tfboard_loggers.TFBoardKerasModelWeightsLogger("3encoder_result/gen")
his_log_dis = tfboard_loggers.TFBoardKerasModelWeightsLogger("3encoder_result/dis")


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

#############################tensorflow 3encoder
def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d", padding='REFLECT'):
    with tf.variable_scope(name):

        # reflect padding
        if padding == 'REFLECT':
            in_height, in_width = input_.get_shape().as_list()[1:3]
            if (in_height % d_h == 0):
                pad_along_height = max(k_h - d_h, 0)
            else:
                pad_along_height = max(k_h - (in_height % d_h), 0)
            if (in_width % d_w == 0):
                pad_along_width = max(k_w - d_w, 0)
            else:
                pad_along_width = max(k_w - (in_width % d_w), 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left

            input_ = tf.pad(input_, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "REFLECT")
            padding = 'VALID'

        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv
def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True, is_training=train, scope=name)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
def conv2d_transpose(input_, output_shape,
                     k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                     name="conv2d_transpose", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


########## Additional Helper Functions ##########
Train =False#True
Free_Size =True#False
test_h =960
test_w =1440
c_dim=3
num_res_blocks = 9
gf_dim=64

def generator_f(image1, image2, image3,train=Train, free_size=Free_Size, batch=batch_size):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        if free_size:
            s_h, s_w = test_h, test_w
        else:
            s_h, s_w = img_rows, img_cols
        s2_h, s4_h, s8_h, s16_h, s2_w, s4_w, s8_w, s16_w = \
            int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16), int(s_w / 2), int(s_w / 4), int(s_w / 8), int(
                s_w / 16)

        def residule_block(x, dim, ks=3, s=1, train=True, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, k_h=ks, k_w=ks, d_h=s, d_w=s, padding='VALID', name=name + '_c1'),
                           train=train, name=name + '_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = batch_norm(conv2d(y, dim, k_h=ks, k_w=ks, d_h=s, d_w=s, padding='VALID', name=name + '_c2'),
                           train=train, name=name + '_bn2')
            return y + x

        print(K.int_shape(image1)[:])
        with tf.variable_scope("encoder1"):
            # image is (256 x 256 x input_c_dim)
            e1_1 = conv2d(image1, gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e1_2 = batch_norm(conv2d(lrelu(e1_1), gf_dim * 2, name='g_e2_conv'), train=train, name='g_e2_bn')

        with tf.variable_scope("encoder2"):
            # image is (256 x 256 x input_c_dim)
            e2_1 = conv2d(image2, gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e2_2 = batch_norm(conv2d(lrelu(e2_1), gf_dim * 2, name='g_e2_conv'), train=train, name='g_e2_bn')

        with tf.variable_scope("encoder3"):
            # image is (256 x 256 x input_c_dim)
            e3_1 = conv2d(image3, gf_dim, name='g_e1_conv')
            # e1 is (128 x 128 x self.gf_dim)
            e3_2 = batch_norm(conv2d(lrelu(e3_1), gf_dim * 2, name='g_e2_conv'), train=train, name='g_e2_bn')

        with tf.variable_scope('merger'):
            e_2 = tf.concat([e1_2, e2_2, e3_2], 3)
            # e2 is (64 x 64 x self.gf_dim*2*3)
            e_3 = batch_norm(conv2d(lrelu(e_2), gf_dim * 4, name='g_e3_conv'), train=train, name='g_e3_bn')
            # e3 is (32 x 32 x self.gf_dim*4)

            res_layer = e_3
            for i in range(num_res_blocks):
                # res_layer = batch_norm(conv2d(lrelu(res_layer), self.gf_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, 
                #                              name='g_e5_conv_%d' %(i+1)), train=train, name='g_e5_bn_%d' %(i+1))
                res_layer = residule_block(tf.nn.relu(res_layer), gf_dim * 4, ks=3, train=train,
                                           name='g_r%d' % (i + 1))

        with tf.variable_scope("decoder"):
            d0 = tf.concat([res_layer, e_3], 3)
            # d0 is (32 x 32 x self.gf_dim*4*2)

            d1 = batch_norm(conv2d_transpose(tf.nn.relu(d0),
                                             [batch, s4_h, s4_w, gf_dim * 2], name='g_d1'), train=train,
                            name='g_d1_bn')
            d1 = tf.concat([d1, e1_2, e2_2, e3_2], 3)
            # d1 is (64 x 64 x self.gf_dim*2*4)

            d2 = batch_norm(conv2d_transpose(tf.nn.relu(d1),
                                             [batch, s2_h, s2_w, gf_dim], name='g_d2'), train=train,
                            name='g_d2_bn')
            d2 = tf.concat([d2, e1_1, e2_1, e3_1], 3)
            # d2 is (128 x 128 x self.gf_dim*1*4)

            d3 = batch_norm(conv2d_transpose(tf.nn.relu(d2),
                                             [batch, s_h, s_w, gf_dim], name='g_d3'), train=train,
                            name='g_d3_bn')
            # d3 is (256 x 256 x self.gf_dim)

            out = conv2d(tf.nn.relu(d3), c_dim, d_h=1, d_w=1, name='g_d_out_conv')
            print(K.int_shape(out)[:])
            return tf.nn.tanh(out)


def generator_model(test=False, attention=False):
    global BATCH_SIZE
    # imgs: input: 256x256xch
    # U-Net structure, must change to relu
    global img_rows, img_cols
    if test:
        img_rows = test_h
        img_cols = test_w
    inputs1 = Input((img_rows, img_cols, IN_CH * 2))
    inputs2 = Input((img_rows, img_cols, IN_CH * 2))
    inputs3 = Input((img_rows, img_cols, IN_CH * 2))

    out = KL.Lambda(lambda x: generator_f(*x), name='crop_local')([inputs1,inputs2,inputs3])
    model = Model(input=[inputs1,inputs2,inputs3], output=out)
    return model
    
    # # image is (256 x 256 x input_c_dim)
    # e1_1 = Convolution2D(gf_dim, 5,5, subsample=(2, 2), init='uniform', border_mode='same')(inputs1)
    # e1_2 = LeakyReLU(alpha=0.2)(e1_1)
    # e1_2 = Convolution2D(gf_dim*2, 5,5, subsample=(2, 2), init='uniform', border_mode='same')(e1_2)
    # # e1 is (128 x 128 x self.gf_dim)
    # e1_2 = BatchNormalization()(e1_2)
    # 
    # 
    # # image is (256 x 256 x input_c_dim)
    # e2_1 = Convolution2D(gf_dim, 5,5, subsample=(2, 2), init='uniform', border_mode='same')(inputs2)
    # e2_2 = LeakyReLU(alpha=0.2)(e2_1)
    # e2_2 = Convolution2D(gf_dim * 2, 5, 5, subsample=(2, 2), init='uniform', border_mode='same')(e2_2)
    # # e1 is (128 x 128 x self.gf_dim)
    # e2_2 = BatchNormalization()(e2_2)
    # 
    # 
    # # image is (256 x 256 x input_c_dim)
    # e3_1 = Convolution2D(gf_dim, 5,5, subsample=(2, 2), init='uniform', border_mode='same')(inputs3)
    # e3_2 = LeakyReLU(alpha=0.2)(e3_1)
    # # e1 is (128 x 128 x self.gf_dim)
    # e3_2 = Convolution2D(gf_dim * 2, 5, 5, subsample=(2, 2), init='uniform', border_mode='same')(e3_2)
    # e3_2 = BatchNormalization()(e3_2)
    # 
    # 
    # e_2 = Concatenate(axis=3)([e1_2, e2_2, e3_2])
    # e_2 = LeakyReLU(alpha=0.2)(e_2)
    # # e2 is (64 x 64 x self.gf_dim*2*3)
    # e_2 = Convolution2D(gf_dim * 4, 5, 5, subsample=(2, 2), init='uniform', border_mode='same')(e_2)
    # e_3 = BatchNormalization()(e_2)
    # # e3 is (32 x 32 x self.gf_dim*4)
    # 
    # res_layer = e_3
    # for i in range(num_res_blocks):
    #     temp = res_layer
    #     res_layer = Convolution2D(gf_dim * 4, 3, 3, subsample=(1, 1),activation='relu', init='uniform', border_mode='same')(temp)
    #     res_layer = BatchNormalization()(res_layer)
    #     res_layer = Convolution2D(gf_dim * 4, 3, 3, subsample=(1, 1), activation='relu', init='uniform',border_mode='same')(res_layer)
    #     res_layer = BatchNormalization()(res_layer)
    #     res_layer = Add()([res_layer,temp])
    # 
    # d0 =  Concatenate(axis=3)([res_layer, e_3])
    # # d0 is (32 x 32 x self.gf_dim*4*2)
    # 
    # d1 = Deconvolution2D(gf_dim*2, 5, 5, subsample=(2,2),  activation='relu',init='uniform', border_mode='same')(d0)
    # d1 = BatchNormalization()(d1)
    # d1 = Concatenate(axis=3)([d1, e1_2, e2_2, e3_2])
    # # d1 is (64 x 64 x self.gf_dim*2*4)
    # 
    # d2 = Deconvolution2D(gf_dim , 5, 5, subsample=(2, 2), activation='relu', init='uniform', border_mode='same')(d1)
    # d2 = BatchNormalization()(d2)  
    # d2 = Concatenate(axis=3)([d2, e1_1, e2_1, e3_1])
    # # d2 is (128 x 128 x self.gf_dim*1*4)
    # 
    # d3 = Deconvolution2D(gf_dim, 5, 5, subsample=(2, 2), activation='relu', init='uniform', border_mode='same')(d2)
    # d3 = BatchNormalization()(d3)
    # # d3 is (256 x 256 x self.gf_dim)
    # 
    # out = Convolution2D(IN_CH, 5, 5, subsample=(1, 1), activation='relu', init='uniform', border_mode='same')(d3)
    # out = Activation('tanh')(out)
    # 
    # model = Model(input=[inputs1,inputs2,inputs3], output=out)
    # return model


def discriminator_model(attention=False):
    """ return a (b, 1) logits"""
    model = Sequential()
    model.add(Convolution2D(64, 4, 4, border_mode='same', input_shape=(img_rows, img_cols, IN_CH * 7)))
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


def generator_containing_discriminator(generator, discriminator, batch):
    inputs1 = Input((img_cols, img_rows, IN_CH*2))
    inputs2 = Input((img_rows, img_cols, IN_CH * 2))
    inputs3 = Input((img_rows, img_cols, IN_CH * 2))

    x_generator = generator([inputs1,inputs2,inputs3])
    inputs10 = KL.Lambda(lambda x: K.reshape(x, (batch, img_rows, img_cols, IN_CH * 2)))(inputs1)
    inputs20 = KL.Lambda(lambda x: K.reshape(x, (batch, img_rows, img_cols, IN_CH * 2)))(inputs2)
    inputs30 = KL.Lambda(lambda x: K.reshape(x, (batch, img_rows, img_cols, IN_CH * 2)))(inputs3)
    merged = Concatenate(axis=3)([inputs10,inputs20,inputs30, x_generator])
    discriminator.trainable = False
    x_discriminator = discriminator(merged)

    model = Model(input=[inputs1,inputs2,inputs3], output=[x_generator, x_discriminator])

    return model


def discriminator_loss(y_true, y_pred):
    global BATCH_SIZE
    return K.mean(K.square(K.flatten(y_pred) - K.concatenate([K.ones_like(K.flatten(y_pred[:BATCH_SIZE, :, :, :])),
                                                              K.zeros_like(K.flatten(y_pred[:BATCH_SIZE, :, :, :]))])),
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


def train(BATCH, epochs=1000, attention=False):
    suffix = ''
    if attention:
        suffix = '_attention'

    sess = K.get_session()
    tfrecord_list = glob(os.path.join(train_path, '**', '*.tfrecords'))
    assert (tfrecord_list)
    shuffle(tfrecord_list)
    filename_queue = tf.train.string_input_producer(tfrecord_list, num_epochs=epochs+1, shuffle=True)
    in_LDRs, in_HDRs, _, ref_HDR, _, _ = load_data(filename_queue, BATCH, img_rows, img_rows, IN_CH,
                                                   num_shots)
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    discriminator = discriminator_model(attention=attention)
    generator = generator_model(attention=attention)
    generator.summary()
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator, BATCH)
    optimizer = Adam(0.0002, 0.5)
    if os.path.isfile('generator_3encoder' + suffix + '.h5'):
        generator.load_weights('generator_3encoder' + suffix + '.h5')
    generator.compile(loss='mse', optimizer=optimizer)
    discriminator_on_generator.compile(loss=[generator_l1_loss, discriminator_on_generator_loss], optimizer=optimizer)
    discriminator.trainable = True
    if os.path.isfile('discriminator_3encoder' + suffix + '.h5'):
        discriminator.load_weights('discriminator_3encoder' + suffix + '.h5')
    discriminator.compile(loss=discriminator_loss, optimizer=optimizer)
    epoch_star_time = time.time()

    tensorboard_2.set_model(discriminator)
    tensorboard_3.set_model(discriminator_on_generator)
    writer = tf.summary.FileWriter('3encoder_result/logs/')

    for epoch in range(epochs):
        print("Epoch : %d/%d" % (epoch, epochs))

        for index in range(steps):
            global step_count
            batch_star_time = time.time()

            image1 = tf.concat([tf.slice(in_LDRs, [0, 0, 0, 0], [-1, -1, -1, IN_CH]),
                                tf.slice(in_HDRs, [0, 0, 0, 0], [-1, -1, -1, IN_CH])], 3)
            image2 = tf.concat([tf.slice(in_LDRs, [0, 0, 0, IN_CH], [-1, -1, -1, IN_CH]),
                                tf.slice(in_HDRs, [0, 0, 0, IN_CH], [-1, -1, -1, IN_CH])], 3)
            image3 = tf.concat([tf.slice(in_LDRs, [0, 0, 0, IN_CH * 2], [-1, -1, -1, IN_CH]),
                                tf.slice(in_HDRs, [0, 0, 0, IN_CH * 2], [-1, -1, -1, IN_CH])], 3)
            X_train1,X_train2,X_train3, Y_train = sess.run([image1,image2,image3, ref_HDR])

            image_batch = tonemap_np(Y_train)
            generated_images = generator.predict([X_train1,X_train2,X_train3])

            real_pairs = np.concatenate((X_train1,X_train2,X_train3, image_batch), axis=3)
            fake_pairs = np.concatenate((X_train1,X_train2,X_train3, generated_images), axis=3)
            X = np.concatenate((real_pairs, fake_pairs))
            y = np.zeros((2 * BATCH, 64, 64, 1))
            d_loss = discriminator.train_on_batch(X, y)
            tensorboard_2.on_epoch_end(step_count, named_logs('d_loss', d_loss))

            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch([X_train1,X_train2,X_train3], [image_batch, np.ones((BATCH, 64, 64, 1))])
            tensorboard_3.on_epoch_end(step_count, named_logs('g_loss', g_loss[1]))
            discriminator.trainable = True

            if step_count % 100 == 0:
                ref = tf.summary.image("G_ref/{}".format(step_count), image_batch, max_outputs=12)
                gen = tf.summary.image("gen/{}".format(step_count), generated_images, max_outputs=12)
                merge_sum = tf.summary.merge([ref, gen])
                sum = sess.run(merge_sum)
                writer.add_summary(sum, step_count)
                writer.flush()
                his_log_dis.log_weights(discriminator, step_count)
                his_log_gen.log_weights(generator, step_count)
            print("epoch:%d, batch:%d/%d, d_loss:%f, g_loss:%f, time:%f" % (
            epoch, index, steps, d_loss, g_loss[1], time.time() - batch_star_time))
            step_count += 1
        generator.save_weights('generator_3encoder' + suffix + '.h5', True)
        discriminator.save_weights('discriminator_3encoder' + suffix + '.h5', True)

    print("epoch_total_time : %f" % (time.time() - epoch_star_time))
    coord.request_stop()
    coord.join(threads)


def test(BATCH, attention=False):
    suffix = ''
    if attention:
        suffix = '_attention'
    generator = generator_model(test=True)
    optimizer = Adam(0.0002, 0.5)
    generator.compile(loss='mse', optimizer=optimizer)
    generator.load_weights('generator_3encoder' + suffix + '.h5')

    scene_dirs = sorted(os.listdir(test_path))
    nScenes = len(scene_dirs)
    num_batch = int(np.ceil(nScenes / BATCH))

    psnr_f = open(os.path.join(result_path, 'psnr.txt'), 'w')
    psnr = []
    for idx in range(0, num_batch):
        print('batch no. %d:' % (idx + 1))
        psnr_f.write('batch no. %d:\n' % (idx + 1))

        l = idx * BATCH
        u = min((idx + 1) * BATCH, nScenes)
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
        if batchSz < BATCH:
            print(batchSz)
            padSz = ((0, int(BATCH - batchSz)), (0, 0), (0, 0), (0, 0))
            batch_in_LDRs = np.pad(batch_in_LDRs, padSz, 'wrap').astype(np.float32)
            batch_in_HDRs = np.pad(batch_in_HDRs, padSz, 'wrap').astype(np.float32)

        image1 = np.concatenate((batch_in_LDRs[:, :, :, 0:IN_CH],batch_in_HDRs[:, :, :, 0:IN_CH]), axis=3)
        image2 = np.concatenate((batch_in_LDRs[:, :, :, IN_CH:IN_CH*2],batch_in_HDRs[:, :, :, IN_CH:IN_CH*2]), axis=3)
        image3 = np.concatenate((batch_in_LDRs[:, :, :, IN_CH*2:IN_CH*3],batch_in_HDRs[:, :, :, IN_CH*2:IN_CH*3]), axis=3)

        st = time.time()
        generated_images = generator.predict([image1,image2,image3])
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
    # train(BATCH=BATCH_SIZE, epochs=7, attention=False)
    test(BATCH=batch_size, attention=False)
