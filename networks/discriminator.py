# from keras.layers import Flatten, Dense, Input, Reshape, merge, Lambda, Concatenate
# from keras.layers.convolutional import Convolution2D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
# from keras.models import Model
# import keras.backend as K
# import numpy as np
# import keras.layers as KL
# import tensorflow.contrib as tf_contrib
# weight_init = tf_contrib.layers.xavier_initializer()
# weight_regularizer = None
# weight_regularizer_fully = None
# import tensorflow as tf
# def PatchGanDiscriminator(output_img_dim, patch_dim, nb_patches):
#     """
#     Creates the generator according to the specs in the paper below.
#     [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
#
#     PatchGAN only penalizes structure at the scale of patches. This
#     discriminator tries to classify if each N x N patch in an
#     image is real or fake. We run this discriminator convolutationally
#     across the image, averaging all responses to provide
#     the ultimate output of D.
#
#     The discriminator has two parts. First part is the actual discriminator
#     seconds part we make it a PatchGAN by running each image patch through the model
#     and then we average the responses
#
#     Discriminator does the following:
#     1. Runs many pieces of the image through the network
#     2. Calculates the cost for each patch
#     3. Returns the avg of the costs as the output of the network
#
#     :param patch_dim: (channels, width, height) T
#     :param nb_patches:
#     :return:
#     """
#     # -------------------------------
#     # DISCRIMINATOR
#     # C64-C128-C256-C512-C512-C512 (for 256x256)
#     # otherwise, it scales from 64
#     # 1 layer block = Conv - BN - LeakyRelu
#     # -------------------------------
#     stride = 2
#     bn_mode = 2
#     axis = 1
#     input_layer = Input(shape=patch_dim)
#
#     # We have to build the discriminator dinamically because
#     # the size of the disc patches is dynamic
#     num_filters_start = 64
#     nb_conv = int(np.floor(np.log(output_img_dim[1]) / np.log(2)))
#     filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]
#
#     # CONV 1
#     # Do first conv bc it is different from the rest
#     # paper skips batch norm for first layer
#     disc_out = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name='disc_conv_1')(input_layer)
#     disc_out = LeakyReLU(alpha=0.2)(disc_out)
#
#     # CONV 2 - CONV N
#     # do the rest of the convs based on the sizes from the filters
#     for i, filter_size in enumerate(filters_list[1:]):
#         name = 'disc_conv_{}'.format(i+2)
#
#         disc_out = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name=name)(disc_out)
#         disc_out = BatchNormalization(momentum=0.9, epsilon=1e-5)(disc_out)
#         disc_out = LeakyReLU(alpha=0.2)(disc_out)
#
#     # ------------------------
#     # BUILD PATCH GAN
#     # this is where we evaluate the loss over each sublayer of the input
#     # ------------------------
#     patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=disc_out,
#                                                       patch_dim=patch_dim,
#                                                       input_layer=input_layer,
#                                                       nb_patches=nb_patches)
#     return patch_gan_discriminator
#
# ########## Additional Helper Functions ##########
# def spectral_norm(w, iteration=1):
#     w_shape = w.shape.as_list()
#     w = tf.reshape(w, [-1, w_shape[-1]])
#
#     u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
#
#     u_hat = u
#     v_hat = None
#     for i in range(iteration):
#         """
#         power iteration
#         Usually iteration = 1 will be enough
#         """
#         v_ = tf.matmul(u_hat, tf.transpose(w))
#         v_hat = tf.nn.l2_normalize(v_)
#
#         u_ = tf.matmul(v_hat, w)
#         u_hat = tf.nn.l2_normalize(u_)
#
#     u_hat = tf.stop_gradient(u_hat)
#     v_hat = tf.stop_gradient(v_hat)
#
#     sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
#
#     with tf.control_dependencies([u.assign(u_hat)]):
#         w_norm = w / sigma
#         w_norm = tf.reshape(w_norm, w_shape)
#
#     return w_norm
#
# def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True,sn=False, scope='conv_0'):
#     #
#     with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
#         if pad > 0:
#             h = x.get_shape().as_list()[1]
#             if h % stride== 0:
#                 pad = pad * 2
#             else:
#                 pad = max(kernel - (h % stride), 0)
#
#             pad_top = pad // 2
#             pad_bottom = pad - pad_top
#             pad_left = pad // 2
#             pad_right = pad - pad_left
#
#             if pad_type == 'zero':
#                 x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
#             if pad_type == 'reflect':
#                 x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
#
#         if sn:
#             w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
#                                 regularizer=weight_regularizer)
#             x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
#                              strides=[1, stride, stride, 1], padding='VALID')
#             if use_bias:
#                 bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
#                 x = tf.nn.bias_add(x, bias)
#
#         else:
#             x = tf.layers.conv2d(inputs=x, filters=channels,
#                                  kernel_size=kernel, kernel_initializer=weight_init,
#                                  kernel_regularizer=weight_regularizer,
#                                  strides=stride, use_bias=use_bias)
#
#         return x
#
# def max_pooling(x, scope='max_pooling') :
#     with tf.variable_scope(scope):
#         y = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#         return y
#
# def hw_flatten(x) :
#     return tf.reshape(x, shape=[-1, x.shape[1] * x.shape[2], x.shape[-1]])
#
#
# def google_attention(x,scope='attention'):
#     with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#         channels = K.int_shape(x)[-1]
#
#         batch_size, height, width, num_channels = x.get_shape().as_list()
#         # f = KL.Lambda(lambda x: conv(*x), arguments={'channels':channels // 8, 'kernel':1,'stride':1, 'sn':True})([x])# [bs, h, w, c']
#         # f = KL.Lambda(lambda x: max_pooling(*x))([f])
#         f = conv(x, channels // 8, kernel=1, stride=1, sn=True,scope='f_conv')  # [bs, h, w, c']
#         print(K.int_shape(f)[:])
#         f = max_pooling(f, scope='f_pooling')
#         g = conv(x, channels // 8, kernel=1, stride=1, sn=True, scope='g_conv')  # [bs, h, w, c']
#
#         h = conv(x, channels // 2, kernel=1, stride=1, sn=True, scope='h_conv')  # [bs, h, w, c]
#         h = max_pooling(h)
#
#         # N = h * w
#         s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
#
#         beta = tf.nn.softmax(s)  # attention map
#
#         o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
#         gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
#
#         o = tf.reshape(o, shape=[-1, height, width, num_channels // 2])  # [bs, h, w, C]
#         o = conv(o, channels, kernel=1, stride=1, sn=True, scope='o_conv')
#         x0 = tf.add(tf.multiply(gamma, o), x)
#
#         return x0
# ########## Additional Helper Functions ##########
#
#
# def PatchGanDiscriminator_attention(output_img_dim, patch_dim, nb_patches):
#     """
#     Creates the generator according to the specs in the paper below.
#     [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
#
#     PatchGAN only penalizes structure at the scale of patches. This
#     discriminator tries to classify if each N x N patch in an
#     image is real or fake. We run this discriminator convolutationally
#     across the image, averaging all responses to provide
#     the ultimate output of D.
#
#     The discriminator has two parts. First part is the actual discriminator
#     seconds part we make it a PatchGAN by running each image patch through the model
#     and then we average the responses
#
#     Discriminator does the following:
#     1. Runs many pieces of the image through the network
#     2. Calculates the cost for each patch
#     3. Returns the avg of the costs as the output of the network
#
#     :param patch_dim: (channels, width, height) T
#     :param nb_patches:
#     :return:
#     """
#     # -------------------------------
#     # DISCRIMINATOR
#     # C64-C128-C256-C512-C512-C512 (for 256x256)
#     # otherwise, it scales from 64
#     # 1 layer block = Conv - BN - LeakyRelu
#     # -------------------------------
#     stride = 2
#     bn_mode = 2
#     axis = 1
#     input_layer = Input(shape=patch_dim)
#
#     # We have to build the discriminator dinamically because
#     # the size of the disc patches is dynamic
#     num_filters_start = 64
#     nb_conv = int(np.floor(np.log(output_img_dim[1]) / np.log(2)))
#     filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]
#
#     # CONV 1
#     # Do first conv bc it is different from the rest
#     # paper skips batch norm for first layer
#     disc_out = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name='disc_conv_1')(input_layer)
#     disc_out = LeakyReLU(alpha=0.2)(disc_out)
#
#     # CONV 2 - CONV N
#     # do the rest of the convs based on the sizes from the filters
#     for i, filter_size in enumerate(filters_list[1:]):
#         name = 'disc_conv_{}'.format(i+2)
#
#         disc_out = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name=name)(disc_out)
#         disc_out = BatchNormalization()(disc_out)
#         disc_out = LeakyReLU(alpha=0.2)(disc_out)
#         if i == 0:
#             disc_out = KL.Lambda(lambda x: google_attention(x), name='crop_local')(disc_out)
#
#     # ------------------------
#     # BUILD PATCH GAN
#     # this is where we evaluate the loss over each sublayer of the input
#     # ------------------------
#     patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=disc_out,
#                                                       patch_dim=patch_dim,
#                                                       input_layer=input_layer,
#                                                       nb_patches=nb_patches)
#     return patch_gan_discriminator
#
#
#
# def generate_patch_gan_loss(last_disc_conv_layer, patch_dim, input_layer, nb_patches):
#
#     # generate a list of inputs for the different patches to the network
#     list_input = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]
#
#     # get an activation
#     x_flat = Flatten()(last_disc_conv_layer)
#     x = Dense(2, activation='sigmoid', name="disc_dense")(x_flat)
#
#     patch_gan = Model(input=[input_layer], output=[x, x_flat], name="patch_gan")
#
#     # generate individual losses for each patch
#     x = [patch_gan(patch)[0] for patch in list_input]
#     x_mbd = [patch_gan(patch)[1] for patch in list_input]
#
#     # merge layers if have multiple patches (aka perceptual loss)
#     if len(x) > 1:
#         x_tem = x[0]
#         for k in range(1, len(x)):
#             x_tem = Concatenate(axis=0)([x_tem, x[k]])
#         x = x_tem
#     else:
#         x = x[0]
#
#     # merge mbd if needed
#     # mbd = mini batch discrimination
#     # https://arxiv.org/pdf/1606.03498.pdf
#     if len(x_mbd) > 1:
#         x_tem = x_mbd[0]
#         for k in range(1, len(x_mbd)):
#             x_tem = Concatenate(axis=0)([x_tem, x_mbd[k]])
#         x_mbd = x_tem
#     else:
#         x_mbd = x_mbd[0]
#
#     num_kernels = 100
#     dim_per_kernel = 5
#
#     M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
#     MBD = Lambda(minb_disc, output_shape=lambda_output)
#
#     x_mbd = M(x_mbd)
#     x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
#     x_mbd = MBD(x_mbd)
#     x = Concatenate(axis=-1)([x, x_mbd])
#
#     x_out = Dense(2, activation="sigmoid", name="disc_output")(x)
#
#     discriminator = Model(input=list_input, output=[x_out], name='discriminator_nn')
#     return discriminator
#
#
# def lambda_output(input_shape):
#     return input_shape[:2]
#
#
# def minb_disc(x):
#     diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
#     abs_diffs = K.sum(K.abs(diffs), 2)
#     x = K.sum(K.exp(-abs_diffs), 2)
#
#     return x
#
#
#
# def PatchGanDiscriminator_g(input_img_dim):
#     """
#     Creates the generator according to the specs in the paper below.
#     [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
#
#     PatchGAN only penalizes structure at the scale of patches. This
#     discriminator tries to classify if each N x N patch in an
#     image is real or fake. We run this discriminator convolutationally
#     across the image, averaging all responses to provide
#     the ultimate output of D.
#
#     The discriminator has two parts. First part is the actual discriminator
#     seconds part we make it a PatchGAN by running each image patch through the model
#     and then we average the responses
#
#     Discriminator does the following:
#     1. Runs many pieces of the image through the network
#     2. Calculates the cost for each patch
#     3. Returns the avg of the costs as the output of the network
#
#     :param patch_dim: (channels, width, height) T
#     :param nb_patches:
#     :return:
#     """
#     # -------------------------------
#     # DISCRIMINATOR
#     # C64-C128-C256-C512-C512-C512 (for 256x256)
#     # otherwise, it scales from 64
#     # 1 layer block = Conv - BN - LeakyRelu
#     # -------------------------------
#     stride = 2
#     bn_mode = 2
#     axis = 1
#     input_layer = Input(shape=input_img_dim)
#
#     # We have to build the discriminator dinamically because
#     # the size of the disc patches is dynamic
#     num_filters_start = 64
#     nb_conv = int(np.floor(np.log(input_img_dim[1]) / np.log(2)))
#     filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]
#
#     # CONV 1
#     # Do first conv bc it is different from the rest
#     # paper skips batch norm for first layer
#     disc_out = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name='disc_conv_1')(input_layer)
#     disc_out = LeakyReLU(alpha=0.2)(disc_out)
#
#     # CONV 2 - CONV N
#     # do the rest of the convs based on the sizes from the filters
#     for i, filter_size in enumerate(filters_list[1:]):
#         name = 'disc_conv_{}'.format(i+2)
#
#         disc_out = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name=name)(disc_out)
#         disc_out = BatchNormalization()(disc_out)
#         disc_out = LeakyReLU(alpha=0.2)(disc_out)
#
#     # ------------------------
#     # BUILD PATCH GAN
#     # this is where we evaluate the loss over each sublayer of the input
#     # ------------------------
#     disc_out = Flatten()(disc_out)
#     x_out = Dense(2, activation="softmax", name="disc_output")(disc_out)
#     discriminator = Model(input=input_layer, output=[x_out], name='discriminator_nn')
#     return discriminator
#


from keras.layers import Flatten, Dense, Input, Reshape, Lambda, Concatenate
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K
import numpy as np

def PatchGanDiscriminator(output_img_dim, patch_dim, nb_patches):
    """
    Creates the generator according to the specs in the paper below.
    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]

    PatchGAN only penalizes structure at the scale of patches. This
    discriminator tries to classify if each N x N patch in an
    image is real or fake. We run this discriminator convolutationally
    across the image, averaging all responses to provide
    the ultimate output of D.

    The discriminator has two parts. First part is the actual discriminator
    seconds part we make it a PatchGAN by running each image patch through the model
    and then we average the responses

    Discriminator does the following:
    1. Runs many pieces of the image through the network
    2. Calculates the cost for each patch
    3. Returns the avg of the costs as the output of the network

    :param patch_dim: (channels, width, height) T
    :param nb_patches:
    :return:
    """
    # -------------------------------
    # DISCRIMINATOR
    # C64-C128-C256-C512-C512-C512 (for 256x256)
    # otherwise, it scales from 64
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    bn_mode = 2
    axis = 1
    input_layer = Input(shape=patch_dim)

    # We have to build the discriminator dinamically because
    # the size of the disc patches is dynamic
    num_filters_start = 64
    nb_conv = int(np.floor(np.log(output_img_dim[1]) / np.log(2)))
    filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]

    # CONV 1
    # Do first conv bc it is different from the rest
    # paper skips batch norm for first layer
    disc_out = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name='disc_conv_1')(input_layer)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # CONV 2 - CONV N
    # do the rest of the convs based on the sizes from the filters
    for i, filter_size in enumerate(filters_list[1:]):
        name = 'disc_conv_{}'.format(i+2)

        disc_out = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name=name)(disc_out)
        disc_out = BatchNormalization(momentum=0.9, epsilon=1e-5)(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # ------------------------
    # BUILD PATCH GAN
    # this is where we evaluate the loss over each sublayer of the input
    # ------------------------
    patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=disc_out,
                                                      patch_dim=patch_dim,
                                                      input_layer=input_layer,
                                                      nb_patches=nb_patches)
    return patch_gan_discriminator



def generate_patch_gan_loss(last_disc_conv_layer, patch_dim, input_layer, nb_patches):

    # generate a list of inputs for the different patches to the network
    list_input = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]

    # get an activation
    x_flat = Flatten()(last_disc_conv_layer)
    x = Dense(1, activation='sigmoid', name="disc_dense")(x_flat)

    patch_gan = Model(input=[input_layer], output=[x], name="patch_gan")

    # generate individual losses for each patch
    x = [patch_gan(patch) for patch in list_input]
    # x_mbd = [patch_gan(patch)[1] for patch in list_input]

    if len(x) > 1:
        x_tem = x[0]
        for k in range(1, len(x)):
            x_tem = Concatenate(axis=0)([x_tem, x[k]])
        x = x_tem
    else:
        x = x[0]

    # merge mbd if needed
    # mbd = mini batch discrimination
    # https://arxiv.org/pdf/1606.03498.pdf
    # if len(x_mbd) > 1:
    #     x_tem = x_mbd[0]
    #     for k in range(1, len(x_mbd)):
    #         x_tem = Concatenate(axis=0)([x_tem, x_mbd[k]])
    #     x_mbd = x_tem
    # else:
    #     x_mbd = x_mbd[0]
    #
    # num_kernels = 100
    # dim_per_kernel = 5
    #
    # M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
    # MBD = Lambda(minb_disc, output_shape=lambda_output)
    #
    # x_mbd = M(x_mbd)
    # x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
    # x_mbd = MBD(x_mbd)
    # x = Concatenate(axis=-1)([x, x_mbd])

    x_out = Dense(1, activation="sigmoid", name="disc_output")(x)

    discriminator = Model(input=list_input, output=[x_out], name='discriminator_nn')
    return discriminator



def lambda_output(input_shape):
    return input_shape[:2]


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x


