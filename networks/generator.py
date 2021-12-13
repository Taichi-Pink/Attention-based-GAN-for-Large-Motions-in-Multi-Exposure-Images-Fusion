# from keras.layers import Activation, Input, Dropout, Concatenate
# from keras.layers.convolutional import Convolution2D, UpSampling2D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
# from keras.models import Model
# import keras.layers as KL
# import tensorflow as tf
# from keras import backend as K
# import numpy as np
# import tensorflow.contrib as tf_contrib
# weight_init = tf_contrib.layers.xavier_initializer()
# weight_regularizer = None
# weight_regularizer_fully = None
# """
# There are two models available for the generator:
# 1. AE Generator
# 2. UNet with skip connections
# """
# 
# 
# def make_generator_ae(input_layer, num_output_filters):
#     """
#     Creates the generator according to the specs in the paper below.
#     [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
#     :param model:
#     :return:
#     """
#     # -------------------------------
#     # ENCODER
#     # C64-C128-C256-C512-C512-C512-C512-C512
#     # 1 layer block = Conv - BN - LeakyRelu
#     # -------------------------------
#     stride = 2
#     filter_sizes = [64, 128, 256, 512, 512, 512, 512, 512]
# 
#     encoder = input_layer
#     for filter_size in filter_sizes:
#         encoder = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(encoder)
#         # paper skips batch norm for first layer
#         if filter_size != 64:
#             encoder = BatchNormalization()(encoder)
#         encoder = Activation(LeakyReLU(alpha=0.2))(encoder)
# 
#     # -------------------------------
#     # DECODER
#     # CD512-CD512-CD512-C512-C512-C256-C128-C64
#     # 1 layer block = Conv - Upsample - BN - DO - Relu
#     # -------------------------------
#     stride = 2
#     filter_sizes = [512, 512, 512, 512, 512, 256, 128, 64]
# 
#     decoder = encoder
#     for filter_size in filter_sizes:
#         decoder = UpSampling2D(size=(2, 2))(decoder)
#         decoder = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same')(decoder)
#         decoder = BatchNormalization()(decoder)
#         decoder = Dropout(p=0.5)(decoder)
#         decoder = Activation('relu')(decoder)
# 
#     # After the last layer in the decoder, a convolution is applied
#     # to map to the number of output channels (3 in general,
#     # except in colorization, where it is 2), followed by a Tanh
#     # function.
#     decoder = Convolution2D(nb_filter=num_output_filters, nb_row=4, nb_col=4, border_mode='same')(decoder)
#     generator = Activation('tanh')(decoder)
#     return generator
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
# def UNETGenerator(input_img_dim, num_output_channels):
#     """
#     Creates the generator according to the specs in the paper below.
#     It's basically a skip layer AutoEncoder
# 
#     Generator does the following:
#     1. Takes in an image
#     2. Generates an image from this image
# 
#     Differs from a standard GAN because the image isn't random.
#     This model tries to learn a mapping from a suboptimal image to an optimal image.
# 
#     [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
#     :param input_img_dim: (channel, height, width)
#     :param output_img_dim: (channel, height, width)
#     :return:
#     """
#     # -------------------------------
#     # ENCODER
#     # C64-C128-C256-C512-C512-C512-C512-C512
#     # 1 layer block = Conv - BN - LeakyRelu
#     # -------------------------------
#     stride = 2
#     merge_mode = 'concat'
# 
#     # batch norm mode
#     bn_mode = 2
# 
#     # batch norm merge axis
#     bn_axis = 1
# 
#     input_layer = Input(shape=input_img_dim, name="unet_input")
# 
#     # 1 encoder C64
#     # skip batchnorm on this layer on purpose (from paper)
#     en_1 = Convolution2D(nb_filter=64, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(input_layer)
#     en_1 = LeakyReLU(alpha=0.2)(en_1)
# 
#     # 2 encoder C128
#     en_2 = Convolution2D(nb_filter=128, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_1)
#     en_2 = BatchNormalization()(en_2)
#     en_2 = LeakyReLU(alpha=0.2)(en_2)
# #####
#     # en_2 = KL.Lambda(lambda x: google_attention(x), name='crop_local')(en_2)
# 
# #####
#     # 3 encoder C256
#     en_3 = Convolution2D(nb_filter=256, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_2)
#     en_3 = BatchNormalization()(en_3)
#     en_3 = LeakyReLU(alpha=0.2)(en_3)
# 
#     # 4 encoder C512
#     en_4 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_3)
#     en_4 = BatchNormalization()(en_4)
#     en_4 = LeakyReLU(alpha=0.2)(en_4)
# 
#     # 5 encoder C512
#     en_5 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_4)
#     en_5 = BatchNormalization()(en_5)
#     en_5 = LeakyReLU(alpha=0.2)(en_5)
# 
#     # 6 encoder C512
#     en_6 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_5)
#     en_6 = BatchNormalization()(en_6)
#     en_6 = LeakyReLU(alpha=0.2)(en_6)
# 
#     # 7 encoder C512
#     en_7 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_6)
#     en_7 = BatchNormalization()(en_7)
#     en_7 = LeakyReLU(alpha=0.2)(en_7)
# 
#     # 8 encoder C512
#     en_8 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_7)
#     en_8 = BatchNormalization()(en_8)
#     en_8 = LeakyReLU(alpha=0.2)(en_8)
# 
#     # -------------------------------
#     # DECODER
#     # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
#     # 1 layer block = Conv - Upsample - BN - DO - Relu
#     # also adds skip connections (merge). Takes input from previous layer matching encoder layer
#     # -------------------------------
#     # 1 decoder CD512 (decodes en_8)
#     de_1 = UpSampling2D(size=(2, 2))(en_8)
#     de_1 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same')(de_1)
#     de_1 = BatchNormalization()(de_1)
#     de_1 = Dropout(p=0.5)(de_1)
#     de_1 = Concatenate(axis=3)([de_1, en_7])
#     de_1 = Activation('relu')(de_1)
# 
#     # 2 decoder CD1024 (decodes en_7)
#     de_2 = UpSampling2D(size=(2, 2))(de_1)
#     de_2 = Convolution2D(nb_filter=1024, nb_row=5, nb_col=5, border_mode='same')(de_2)
#     de_2 = BatchNormalization()(de_2)
#     de_2 = Dropout(p=0.5)(de_2)
#     de_2 = Concatenate(axis=3)([de_2, en_6])
#     de_2 = Activation('relu')(de_2)
# 
#     # 3 decoder CD1024 (decodes en_6)
#     de_3 = UpSampling2D(size=(2, 2))(de_2)
#     de_3 = Convolution2D(nb_filter=1024, nb_row=5, nb_col=5, border_mode='same')(de_3)
#     de_3 = BatchNormalization()(de_3)
#     de_3 = Dropout(p=0.5)(de_3)
#     de_3 = Concatenate(axis=3)([de_3, en_5])
#     de_3 = Activation('relu')(de_3)
# 
#     # 4 decoder CD1024 (decodes en_5)
#     de_4 = UpSampling2D(size=(2, 2))(de_3)
#     de_4 = Convolution2D(nb_filter=1024, nb_row=5, nb_col=5, border_mode='same')(de_4)
#     de_4 = BatchNormalization()(de_4)
#     de_4 = Dropout(p=0.5)(de_4)
#     de_4 = Concatenate(axis=3)([de_4, en_4])
#     de_4 = Activation('relu')(de_4)
# 
#     # 5 decoder CD1024 (decodes en_4)
#     de_5 = UpSampling2D(size=(2, 2))(de_4)
#     de_5 = Convolution2D(nb_filter=1024, nb_row=5, nb_col=5, border_mode='same')(de_5)
#     de_5 = BatchNormalization()(de_5)
#     de_5 = Dropout(p=0.5)(de_5)
#     de_5 = Concatenate(axis=3)([de_5, en_3])
#     de_5 = Activation('relu')(de_5)
# 
#     # 6 decoder C512 (decodes en_3)
#     de_6 = UpSampling2D(size=(2, 2))(de_5)
#     de_6 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same')(de_6)
#     de_6 = BatchNormalization()(de_6)
#     de_6 = Dropout(p=0.5)(de_6)
#     de_6 = Concatenate(axis=3)([de_6, en_2])
#     de_6 = Activation('relu')(de_6)
# 
#     # 7 decoder CD256 (decodes en_2)
#     de_7 = UpSampling2D(size=(2, 2))(de_6)
#     de_7 = Convolution2D(nb_filter=256, nb_row=5, nb_col=5, border_mode='same')(de_7)
#     de_7 = BatchNormalization()(de_7)
#     de_7 = Dropout(p=0.5)(de_7)
#     de_7 = Concatenate(axis=3)([de_7, en_1])
#     de_7 = Activation('relu')(de_7)
# 
#     # After the last layer in the decoder, a convolution is applied
#     # to map to the number of output channels (3 in general,
#     # except in colorization, where it is 2), followed by a Tanh
#     # function.
#     de_8 = UpSampling2D(size=(2, 2))(de_7)
#     de_8 = Convolution2D(nb_filter=num_output_channels, nb_row=5, nb_col=5, border_mode='same')(de_8)
#     de_8 = Activation('tanh')(de_8)
# 
#     unet_generator = Model(input=[input_layer], output=[de_8], name='unet_generator')
#     return unet_generator
# 
# def UNETGenerator0(input_img_dim, num_output_channels):
#     """
#     Creates the generator according to the specs in the paper below.
#     It's basically a skip layer AutoEncoder
# 
#     Generator does the following:
#     1. Takes in an image
#     2. Generates an image from this image
# 
#     Differs from a standard GAN because the image isn't random.
#     This model tries to learn a mapping from a suboptimal image to an optimal image.
# 
#     [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
#     :param input_img_dim: (channel, height, width)
#     :param output_img_dim: (channel, height, width)
#     :return:
#     """
#     # -------------------------------
#     # ENCODER
#     # C64-C128-C256-C512-C512-C512-C512-C512
#     # 1 layer block = Conv - BN - LeakyRelu
#     # -------------------------------
#     stride = 2
#     merge_mode = 'concat'
# 
#     # batch norm mode
#     bn_mode = 2
# 
#     # batch norm merge axis
#     bn_axis = 1
# 
#     input_layer = Input(shape=input_img_dim, name="unet_input")
# 
#     # 1 encoder C64
#     # skip batchnorm on this layer on purpose (from paper)
#     en_1 = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(input_layer)
#     en_1 = LeakyReLU(alpha=0.2)(en_1)
# 
#     # 2 encoder C128
#     en_2 = Convolution2D(nb_filter=128, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_1)
#     en_2 = BatchNormalization()(en_2)
#     en_2 = LeakyReLU(alpha=0.2)(en_2)
# 
# #####
#     # 3 encoder C256
#     en_3 = Convolution2D(nb_filter=256, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_2)
#     en_3 = BatchNormalization()(en_3)
#     en_3 = LeakyReLU(alpha=0.2)(en_3)
# 
#     # 4 encoder C512
#     en_4 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_3)
#     en_4 = BatchNormalization()(en_4)
#     en_4 = LeakyReLU(alpha=0.2)(en_4)
# 
#     # 5 encoder C512
#     en_5 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_4)
#     en_5 = BatchNormalization()(en_5)
#     en_5 = LeakyReLU(alpha=0.2)(en_5)
# 
#     # 6 encoder C512
#     en_6 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_5)
#     en_6 = BatchNormalization()(en_6)
#     en_6 = LeakyReLU(alpha=0.2)(en_6)
# 
#     # 7 encoder C512
#     en_7 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_6)
#     en_7 = BatchNormalization()(en_7)
#     en_7 = LeakyReLU(alpha=0.2)(en_7)
# 
#     # 8 encoder C512
#     en_8 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_7)
#     en_8 = BatchNormalization()(en_8)
#     en_8 = LeakyReLU(alpha=0.2)(en_8)
# 
#     # -------------------------------
#     # DECODER
#     # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
#     # 1 layer block = Conv - Upsample - BN - DO - Relu
#     # also adds skip connections (merge). Takes input from previous layer matching encoder layer
#     # -------------------------------
#     # 1 decoder CD512 (decodes en_8)
#     de_1 = UpSampling2D(size=(2, 2))(en_8)
#     de_1 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same')(de_1)
#     de_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(de_1)
#     de_1 = Dropout(p=0.5)(de_1)
#     de_1 = Concatenate(axis=3)([de_1, en_7])
#     de_1 = Activation('relu')(de_1)
# 
#     # 2 decoder CD1024 (decodes en_7)
#     de_2 = UpSampling2D(size=(2, 2))(de_1)
#     de_2 = Convolution2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_2)
#     de_2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(de_2)
#     de_2 = Dropout(p=0.5)(de_2)
#     de_2 = Concatenate(axis=3)([de_2, en_6])
#     de_2 = Activation('relu')(de_2)
# 
#     # 3 decoder CD1024 (decodes en_6)
#     de_3 = UpSampling2D(size=(2, 2))(de_2)
#     de_3 = Convolution2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_3)
#     de_3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(de_3)
#     de_3 = Dropout(p=0.5)(de_3)
#     de_3 = Concatenate(axis=3)([de_3, en_5])
#     de_3 = Activation('relu')(de_3)
# 
#     # 4 decoder CD1024 (decodes en_5)
#     de_4 = UpSampling2D(size=(2, 2))(de_3)
#     de_4 = Convolution2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_4)
#     de_4 = BatchNormalization(momentum=0.9, epsilon=1e-5)(de_4)
#     de_4 = Dropout(p=0.5)(de_4)
#     de_4 = Concatenate(axis=3)([de_4, en_4])
#     de_4 = Activation('relu')(de_4)
# 
#     # 5 decoder CD1024 (decodes en_4)
#     de_5 = UpSampling2D(size=(2, 2))(de_4)
#     de_5 = Convolution2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_5)
#     de_5 = BatchNormalization(momentum=0.9, epsilon=1e-5)(de_5)
#     de_5 = Dropout(p=0.5)(de_5)
#     de_5 = Concatenate(axis=3)([de_5, en_3])
#     de_5 = Activation('relu')(de_5)
# 
#     # 6 decoder C512 (decodes en_3)
#     de_6 = UpSampling2D(size=(2, 2))(de_5)
#     de_6 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same')(de_6)
#     de_6 = BatchNormalization(momentum=0.9, epsilon=1e-5)(de_6)
#     de_6 = Dropout(p=0.5)(de_6)
#     de_6 = Concatenate(axis=3)([de_6, en_2])
#     de_6 = Activation('relu')(de_6)
# 
#     # 7 decoder CD256 (decodes en_2)
#     de_7 = UpSampling2D(size=(2, 2))(de_6)
#     de_7 = Convolution2D(nb_filter=256, nb_row=4, nb_col=4, border_mode='same')(de_7)
#     de_7 = BatchNormalization(momentum=0.9, epsilon=1e-5)(de_7)
#     de_7 = Dropout(p=0.5)(de_7)
#     de_7 = Concatenate(axis=3)([de_7, en_1])
#     de_7 = Activation('relu')(de_7)
# 
#     # After the last layer in the decoder, a convolution is applied
#     # to map to the number of output channels (3 in general,
#     # except in colorization, where it is 2), followed by a Tanh
#     # function.
#     de_8 = UpSampling2D(size=(2, 2))(de_7)
#     de_8 = Convolution2D(nb_filter=num_output_channels, nb_row=4, nb_col=4, border_mode='same')(de_8)
#     de_8 = Activation('tanh')(de_8)
# 
#     unet_generator = Model(input=[input_layer], output=[de_8], name='unet_generator')
#     return unet_generator
# 
# #############################tensorflow 3encoder
# def conv2d(input_, output_dim,
#            k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#            name="conv2d", padding='REFLECT'):
#     with tf.variable_scope(name):
# 
#         # reflect padding
#         if padding == 'REFLECT':
#             in_height, in_width = input_.get_shape().as_list()[1:3]
#             if (in_height % d_h == 0):
#                 pad_along_height = max(k_h - d_h, 0)
#             else:
#                 pad_along_height = max(k_h - (in_height % d_h), 0)
#             if (in_width % d_w == 0):
#                 pad_along_width = max(k_w - d_w, 0)
#             else:
#                 pad_along_width = max(k_w - (in_width % d_w), 0)
#             pad_top = pad_along_height // 2
#             pad_bottom = pad_along_height - pad_top
#             pad_left = pad_along_width // 2
#             pad_right = pad_along_width - pad_left
# 
#             input_ = tf.pad(input_, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "REFLECT")
#             padding = 'VALID'
# 
#         w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#                             initializer=tf.truncated_normal_initializer(stddev=stddev))
#         conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
# 
#         biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
#         # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
#         conv = tf.nn.bias_add(conv, biases)
# 
#         return conv
# def batch_norm(x, momentum=0.9, epsilon=1e-5, train=True, name="batch_norm"):
#     return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None, epsilon=epsilon, scale=True, is_training=train, scope=name)
# 
# def lrelu(x, leak=0.2, name="lrelu"):
#     with tf.variable_scope(name):
#         f1 = 0.5 * (1 + leak)
#         f2 = 0.5 * (1 - leak)
#         return f1 * x + f2 * abs(x)
# def conv2d_transpose(input_, output_shape,
#                      k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
#                      name="conv2d_transpose", with_w=False):
#     with tf.variable_scope(name):
#         # filter : [height, width, output_channels, in_channels]
#         w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
#                             initializer=tf.random_normal_initializer(stddev=stddev))
# 
#         try:
#             deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
#                                 strides=[1, d_h, d_w, 1])
# 
#         # Support for verisons of TensorFlow before 0.7.0
#         except AttributeError:
#             deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
#                                 strides=[1, d_h, d_w, 1])
# 
#         biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#         # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
#         deconv = tf.nn.bias_add(deconv, biases)
# 
#         if with_w:
#             return deconv, w, biases
#         else:
#             return deconv
# 
# 
# ########## Additional Helper Functions ##########
# 
# Train =True
# Free_Size =False
# batch_size = 1
# 
# test_h =960
# test_w =1440
# c_dim=3
# num_res_blocks = 9
# gf_dim=64
# img_rows = img_cols = 256
# IN_CH = 3
# def generator_f(image1, image2, image3,train=Train, free_size=Free_Size, batch=batch_size):
#     with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
#         if free_size:
#             s_h, s_w = test_h, test_w
#         else:
#             s_h, s_w = img_rows, img_cols
#         s2_h, s4_h, s8_h, s16_h, s2_w, s4_w, s8_w, s16_w = \
#             int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16), int(s_w / 2), int(s_w / 4), int(s_w / 8), int(
#                 s_w / 16)
# 
#         def residule_block(x, dim, ks=3, s=1, train=True, name='res'):
#             p = int((ks - 1) / 2)
#             y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
#             y = batch_norm(conv2d(y, dim, k_h=ks, k_w=ks, d_h=s, d_w=s, padding='VALID', name=name + '_c1'),
#                            train=train, name=name + '_bn1')
#             y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
#             y = batch_norm(conv2d(y, dim, k_h=ks, k_w=ks, d_h=s, d_w=s, padding='VALID', name=name + '_c2'),
#                            train=train, name=name + '_bn2')
#             return y + x
# 
#         print(K.int_shape(image1)[:])
#         with tf.variable_scope("encoder1"):
#             # image is (256 x 256 x input_c_dim)
#             e1_1 = conv2d(image1, gf_dim, name='g_e1_conv')
#             # e1 is (128 x 128 x self.gf_dim)
#             e1_2 = batch_norm(conv2d(lrelu(e1_1), gf_dim * 2, name='g_e2_conv'), train=train, name='g_e2_bn')
# 
#         with tf.variable_scope("encoder2"):
#             # image is (256 x 256 x input_c_dim)
#             e2_1 = conv2d(image2, gf_dim, name='g_e1_conv')
#             # e1 is (128 x 128 x self.gf_dim)
#             e2_2 = batch_norm(conv2d(lrelu(e2_1), gf_dim * 2, name='g_e2_conv'), train=train, name='g_e2_bn')
# 
#         with tf.variable_scope("encoder3"):
#             # image is (256 x 256 x input_c_dim)
#             e3_1 = conv2d(image3, gf_dim, name='g_e1_conv')
#             # e1 is (128 x 128 x self.gf_dim)
#             e3_2 = batch_norm(conv2d(lrelu(e3_1), gf_dim * 2, name='g_e2_conv'), train=train, name='g_e2_bn')
# 
#         with tf.variable_scope('merger'):
#             e_2 = tf.concat([e1_2, e2_2, e3_2], 3)
#             # e2 is (64 x 64 x self.gf_dim*2*3)
#             e_3 = batch_norm(conv2d(lrelu(e_2), gf_dim * 4, name='g_e3_conv'), train=train, name='g_e3_bn')
#             # e3 is (32 x 32 x self.gf_dim*4)
# 
#             res_layer = e_3
#             for i in range(num_res_blocks):
#                 # res_layer = batch_norm(conv2d(lrelu(res_layer), self.gf_dim*4, k_h=3, k_w=3, d_h=1, d_w=1, 
#                 #                              name='g_e5_conv_%d' %(i+1)), train=train, name='g_e5_bn_%d' %(i+1))
#                 res_layer = residule_block(tf.nn.relu(res_layer), gf_dim * 4, ks=3, train=train,
#                                            name='g_r%d' % (i + 1))
# 
#         with tf.variable_scope("decoder"):
#             d0 = tf.concat([res_layer, e_3], 3)
#             # d0 is (32 x 32 x self.gf_dim*4*2)
# 
#             d1 = batch_norm(conv2d_transpose(tf.nn.relu(d0),
#                                              [batch, s4_h, s4_w, gf_dim * 2], name='g_d1'), train=train,
#                             name='g_d1_bn')
#             d1 = tf.concat([d1, e1_2, e2_2, e3_2], 3)
#             # d1 is (64 x 64 x self.gf_dim*2*4)
# 
#             d2 = batch_norm(conv2d_transpose(tf.nn.relu(d1),
#                                              [batch, s2_h, s2_w, gf_dim], name='g_d2'), train=train,
#                             name='g_d2_bn')
#             d2 = tf.concat([d2, e1_1, e2_1, e3_1], 3)
#             # d2 is (128 x 128 x self.gf_dim*1*4)
# 
#             d3 = batch_norm(conv2d_transpose(tf.nn.relu(d2),
#                                              [batch, s_h, s_w, gf_dim], name='g_d3'), train=train,
#                             name='g_d3_bn')
#             # d3 is (256 x 256 x self.gf_dim)
# 
#             out = conv2d(tf.nn.relu(d3), c_dim, d_h=1, d_w=1, name='g_d_out_conv')
#             print(K.int_shape(out)[:])
#             return tf.nn.tanh(out)
# 
# 
# def Encoders(test=False):
#     global BATCH_SIZE
#     # imgs: input: 256x256xch
#     # U-Net structure, must change to relu
#     global img_rows, img_cols
#     if test:
#         img_rows = test_h
#         img_cols = test_w
#     inputs1 = Input((img_rows, img_cols, IN_CH))
#     inputs2 = Input((img_rows, img_cols, IN_CH ))
#     inputs3 = Input((img_rows, img_cols, IN_CH ))
# 
#     out = KL.Lambda(lambda x: generator_f(*x), name='crop_local')([inputs1,inputs2,inputs3])
#     model = Model(input=[inputs1,inputs2,inputs3], output=out)
#     return model

from keras.layers import Activation, Input, Dropout, Concatenate
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

"""
There are two models available for the generator:
1. AE Generator
2. UNet with skip connections
"""


def make_generator_ae(input_layer, num_output_filters):
    """
    Creates the generator according to the specs in the paper below.
    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
    :param model:
    :return:
    """
    # -------------------------------
    # ENCODER
    # C64-C128-C256-C512-C512-C512-C512-C512
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    filter_sizes = [64, 128, 256, 512, 512, 512, 512, 512]

    encoder = input_layer
    for filter_size in filter_sizes:
        encoder = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(encoder)
        # paper skips batch norm for first layer
        if filter_size != 64:
            encoder = BatchNormalization()(encoder)
        encoder = Activation(LeakyReLU(alpha=0.2))(encoder)

    # -------------------------------
    # DECODER
    # CD512-CD512-CD512-C512-C512-C256-C128-C64
    # 1 layer block = Conv - Upsample - BN - DO - Relu
    # -------------------------------
    stride = 2
    filter_sizes = [512, 512, 512, 512, 512, 256, 128, 64]

    decoder = encoder
    for filter_size in filter_sizes:
        decoder = UpSampling2D(size=(2, 2))(decoder)
        decoder = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(p=0.5)(decoder)
        decoder = Activation('relu')(decoder)

    # After the last layer in the decoder, a convolution is applied
    # to map to the number of output channels (3 in general,
    # except in colorization, where it is 2), followed by a Tanh
    # function.
    decoder = Convolution2D(nb_filter=num_output_filters, nb_row=4, nb_col=4, border_mode='same')(decoder)
    generator = Activation('tanh')(decoder)
    return generator


def UNETGenerator(input_img_dim, num_output_channels):
    """
    Creates the generator according to the specs in the paper below.
    It's basically a skip layer AutoEncoder

    Generator does the following:
    1. Takes in an image
    2. Generates an image from this image

    Differs from a standard GAN because the image isn't random.
    This model tries to learn a mapping from a suboptimal image to an optimal image.

    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
    :param input_img_dim: (channel, height, width)
    :param output_img_dim: (channel, height, width)
    :return:
    """
    # -------------------------------
    # ENCODER
    # C64-C128-C256-C512-C512-C512-C512-C512
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    merge_mode = 'concat'

    # batch norm mode
    bn_mode = 2

    # batch norm merge axis
    bn_axis = 1

    input_layer = Input(shape=input_img_dim, name="unet_input")

    # 1 encoder C64
    # skip batchnorm on this layer on purpose (from paper)
    en_1 = Convolution2D(nb_filter=64, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(input_layer)
    en_1 = LeakyReLU(alpha=0.2)(en_1)

    # 2 encoder C128
    en_2 = Convolution2D(nb_filter=128, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_1)
    en_2 = BatchNormalization()(en_2)
    en_2 = LeakyReLU(alpha=0.2)(en_2)

    # 3 encoder C256
    en_3 = Convolution2D(nb_filter=256, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_2)
    en_3 = BatchNormalization()(en_3)
    en_3 = LeakyReLU(alpha=0.2)(en_3)

    # 4 encoder C512
    en_4 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_3)
    en_4 = BatchNormalization()(en_4)
    en_4 = LeakyReLU(alpha=0.2)(en_4)

    # 5 encoder C512
    en_5 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_4)
    en_5 = BatchNormalization()(en_5)
    en_5 = LeakyReLU(alpha=0.2)(en_5)

    # 6 encoder C512
    en_6 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_5)
    en_6 = BatchNormalization()(en_6)
    en_6 = LeakyReLU(alpha=0.2)(en_6)

    # 7 encoder C512
    en_7 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_6)
    en_7 = BatchNormalization()(en_7)
    en_7 = LeakyReLU(alpha=0.2)(en_7)

    # 8 encoder C512
    en_8 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same', subsample=(stride, stride))(en_7)
    en_8 = BatchNormalization()(en_8)
    en_8 = LeakyReLU(alpha=0.2)(en_8)

    # -------------------------------
    # DECODER
    # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    # 1 layer block = Conv - Upsample - BN - DO - Relu
    # also adds skip connections (merge). Takes input from previous layer matching encoder layer
    # -------------------------------
    # 1 decoder CD512 (decodes en_8)
    de_1 = UpSampling2D(size=(2, 2))(en_8)
    de_1 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same')(de_1)
    de_1 = BatchNormalization()(de_1)
    de_1 = Dropout(p=0.5)(de_1)
    de_1 = Concatenate(axis=3)([de_1, en_7])
    de_1 = Activation('relu')(de_1)

    # 2 decoder CD1024 (decodes en_7)
    de_2 = UpSampling2D(size=(2, 2))(de_1)
    de_2 = Convolution2D(nb_filter=1024, nb_row=5, nb_col=5, border_mode='same')(de_2)
    de_2 = BatchNormalization()(de_2)
    de_2 = Dropout(p=0.5)(de_2)
    de_2 = Concatenate(axis=3)([de_2, en_6])
    de_2 = Activation('relu')(de_2)

    # 3 decoder CD1024 (decodes en_6)
    de_3 = UpSampling2D(size=(2, 2))(de_2)
    de_3 = Convolution2D(nb_filter=1024, nb_row=5, nb_col=5, border_mode='same')(de_3)
    de_3 = BatchNormalization()(de_3)
    de_3 = Dropout(p=0.5)(de_3)
    de_3 = Concatenate(axis=3)([de_3, en_5])
    de_3 = Activation('relu')(de_3)

    # 4 decoder CD1024 (decodes en_5)
    de_4 = UpSampling2D(size=(2, 2))(de_3)
    de_4 = Convolution2D(nb_filter=1024, nb_row=5, nb_col=5, border_mode='same')(de_4)
    de_4 = BatchNormalization()(de_4)
    de_4 = Dropout(p=0.5)(de_4)
    de_4 = Concatenate(axis=3)([de_4, en_4])
    de_4 = Activation('relu')(de_4)

    # 5 decoder CD1024 (decodes en_4)
    de_5 = UpSampling2D(size=(2, 2))(de_4)
    de_5 = Convolution2D(nb_filter=1024, nb_row=5, nb_col=5, border_mode='same')(de_5)
    de_5 = BatchNormalization()(de_5)
    de_5 = Dropout(p=0.5)(de_5)
    de_5 = Concatenate(axis=3)([de_5, en_3])
    de_5 = Activation('relu')(de_5)

    # 6 decoder C512 (decodes en_3)
    de_6 = UpSampling2D(size=(2, 2))(de_5)
    de_6 = Convolution2D(nb_filter=512, nb_row=5, nb_col=5, border_mode='same')(de_6)
    de_6 = BatchNormalization()(de_6)
    de_6 = Dropout(p=0.5)(de_6)
    de_6 = Concatenate(axis=3)([de_6, en_2])
    de_6 = Activation('relu')(de_6)

    # 7 decoder CD256 (decodes en_2)
    de_7 = UpSampling2D(size=(2, 2))(de_6)
    de_7 = Convolution2D(nb_filter=256, nb_row=5, nb_col=5, border_mode='same')(de_7)
    de_7 = BatchNormalization()(de_7)
    de_7 = Dropout(p=0.5)(de_7)
    de_7 = Concatenate(axis=3)([de_7, en_1])
    de_7 = Activation('relu')(de_7)

    # After the last layer in the decoder, a convolution is applied
    # to map to the number of output channels (3 in general,
    # except in colorization, where it is 2), followed by a Tanh
    # function.
    de_8 = UpSampling2D(size=(2, 2))(de_7)
    de_8 = Convolution2D(nb_filter=num_output_channels, nb_row=5, nb_col=5, border_mode='same')(de_8)
    de_8 = Activation('tanh')(de_8)

    unet_generator = Model(input=[input_layer], output=[de_8], name='unet_generator')
    return unet_generator