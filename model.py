import tensorflow as tf
import numpy as np
import cv2, glob, os,math
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers, Model, initializers
from tensorflow import keras
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

patch_size = 256
ch         = 3

select_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(patch_size, patch_size, ch))
vgg16.trainable = False
for l in vgg16.layers:
  l.trainable = False
select = [vgg16.get_layer(name).output for name in select_layers]

model_vgg = Model(inputs=vgg16.input, outputs=select)
model_vgg.trainable = False
    
def vgg_loss(y_true, y_pred):
    
    out_pred = model_vgg(y_pred) 
    out_true = model_vgg(y_true)
    loss_f = 0
    for f_g, f_l in zip(out_pred, out_true):
        loss_f = loss_f + K.mean(K.abs(f_g - f_l))
    return loss_f + tf.math.reduce_mean(tf.square(y_true - y_pred))

def generator_model(w,h,c, k1=1, k2=3, s1=1, s2=2, f1=256, f2=64, pad="same", act='relu'):
    input_ = keras.Input((w, h, c))
    e1 = layers.BatchNormalization()(input_)
    e1 = layers.Conv2D(64, k2, strides=(2,2),  padding='same')(e1)
    e1 = layers.LeakyReLU(alpha=0.02)(e1)
    e1 = layers.BatchNormalization()(e1)
    e2 = layers.Conv2D(128, k2, strides=(2,2),  padding='same')(e1)
    e2 = layers.LeakyReLU(alpha=0.02)(e2)
    e2 = layers.BatchNormalization()(e2)
    
    e3 = layers.Conv2D(256, k2, strides=(2,2), padding='same')(e2)
    e3 = layers.LeakyReLU(alpha=0.02)(e3)
    e3 = layers.BatchNormalization()(e3)
    e4 = layers.Conv2D(512, k2, strides=(2,2),  padding='same')(e3)
    e4 = layers.LeakyReLU(alpha=0.02)(e4)
    e4 = layers.BatchNormalization()(e4)
    
    e5 = layers.Conv2D(512, k2, strides=(2,2), padding='same')(e4)
    e5 = layers.LeakyReLU(alpha=0.02)(e5)
    e5 = layers.BatchNormalization()(e5)
    e6 = layers.Conv2D(512, k2, strides=(2,2), padding='same')(e5)
    e6 = layers.LeakyReLU(alpha=0.02)(e6)
    e6 = layers.BatchNormalization()(e6)

    e7 = layers.Conv2D(512, k2, strides=(2,2), padding='same')(e6)
    e7 = layers.LeakyReLU(alpha=0.02)(e7)
    e7 = layers.BatchNormalization()(e7)
    e8 = layers.Conv2D(512, k2, strides=(2,2), padding='same')(e7)
    e8 = layers.LeakyReLU(alpha=0.02)(e8)
    e8 = layers.BatchNormalization()(e8)

    d1 = layers.UpSampling2D(interpolation='nearest')(e8)
    d1 = layers.Conv2D(512, 3, strides=(1, 1), activation='relu',padding='same')(d1)
    d1 = layers.Concatenate(axis=3)([d1, e7])
    d1 = layers.BatchNormalization()(d1)

    d2 = layers.UpSampling2D(interpolation='nearest')(d1)
    d2 = layers.Conv2D(512, 3, strides=(1, 1), activation='relu', padding='same')(d2)
    d2 = layers.Concatenate(axis=3)([d2, e6])
    d2 = layers.BatchNormalization()(d2)

    d3 = layers.Dropout(0.2)(d2)
    d3 = layers.UpSampling2D(interpolation='nearest')(d3)
    d3 = layers.Conv2D(512, 3, strides=(1, 1), activation='relu', padding='same')(d3)
    d3 = layers.Concatenate(axis=3)([d3, e5])
    d3 = layers.BatchNormalization()(d3)

    d4 = layers.Dropout(0.2)(d3)
    d4 = layers.UpSampling2D(interpolation='nearest')(d4)
    d4 = layers.Conv2D(512, 3, strides=(1, 1), activation='relu', padding='same')(d4)
    d4 = layers.Concatenate(axis=3)([d4, e4])
    d4 = layers.BatchNormalization()(d4)

    d5 = layers.Dropout(0.2)(d4)
    d5 = layers.UpSampling2D(interpolation='nearest')(d5)
    d5 = layers.Conv2D(256, 3, strides=(1, 1), activation='relu', padding='same')(d5)
    d5 = layers.Concatenate(axis=3)([d5, e3])
    d5 = layers.BatchNormalization()(d5)

    d6 = layers.Dropout(0.2)(d5)
    d6 = layers.UpSampling2D(interpolation='nearest')(d6)
    d6 = layers.Conv2D(128, 3, strides=(1, 1), activation='relu', padding='same')(d6)
    d6 = layers.Concatenate(axis=3)([d6, e2])
    d6 = layers.BatchNormalization()(d6)

    d7 = layers.Dropout(0.2)(d6)
    d7 = layers.UpSampling2D(interpolation='nearest')(d7)
    d7 = layers.Conv2D(64, 3, strides=(1, 1), activation='relu', padding='same')(d7)
    d7 = layers.Concatenate(axis=3)([d7, e1])

    d7 = layers.BatchNormalization()(d7)
    d8 = layers.UpSampling2D(interpolation='nearest')(d7)
    d9 = layers.Conv2D(3, 3, strides=(1, 1), activation='tanh', padding='same')(d8)

    model = Model(inputs=input_, outputs=d9)
    return model

def discriminator_model(w,h,c, k1=1, k2=3, s1=1, s2=4, f1=256, f2=64, pad="same", act='relu'):
    model = Sequential()
    model.add(layers.Conv2D(64, 4, padding='same',input_shape=(w, h, c)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, 4, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(512, 4, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('tanh'))
    model.add(layers.Conv2D(1, 4, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def generator_containing_discriminator(w,h,c, generator, discriminator):
    input_ = keras.Input((w, h, c))
    out_1  = generator(input_)
    merged = layers.Concatenate(axis=3)([input_, out_1])
    not_trainable(discriminator)
    out_2 = discriminator(merged)
    model = Model(inputs=input_, outputs=[out_1, out_2])
    return model

def not_trainable(model):
    model.trainable = False
    for l in model.layers:
      l.trainable = False

def is_trainable(model):
    model.trainable = True
    for l in model.layers:
      l.trainable = True
def norm_0_to_1(img):
    img = np.float32(img)
    img_flat = img.flatten()
    max_value = np.max(img_flat)
    min_value = np.min(img_flat)
    new_img = (img - min_value) * 1 / (max_value - min_value)
    return new_img