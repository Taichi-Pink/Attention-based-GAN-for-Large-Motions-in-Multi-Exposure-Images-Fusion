import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import matplotlib.pyplot as plt
import math, cv2, glob, time
import numpy as np
import imageio as io
from load_data import load_data
from model import *

patch_size = 256
epochs     = 5
bs         = 1
ch         = 3

def train(gen, dis, dis_on_gen, save_path):
    dataset = load_data(bs=1, patch_size=256)
    print(gen.summary())
    print(dis.summary())
    epoch_star_time = time.time()
    loss_history_d = []
    loss_history_g = []
    for epoch in range(epochs):
        loss_d = []
        loss_g = []
        step  = 0
        for ldr, hdr in dataset:
            ldr = (tf.image.resize_images(tf.cast(ldr, tf.float32), (patch_size, patch_size))/255.0)*2.0-1.0
            hdr = tf.image.resize_images(tf.cast(hdr, tf.float32), (patch_size, patch_size))*2.0-1.0
            
            is_trainable(dis)
            not_trainable(gen)
            generated_images = gen.predict(ldr)

            real_pairs = np.concatenate((ldr,hdr),axis=3)
            fake_pairs = np.concatenate((ldr,generated_images),axis=3)
            x = np.concatenate((real_pairs,fake_pairs))
            #y = np.concatenate((np.ones((bs, patch_size//4, patch_size//4, 1)), np.zeros((bs, patch_size//4, patch_size//4,1))))
            y = np.concatenate((np.ones((bs, 1)), np.zeros((bs,1))))
            d_loss = dis.train_on_batch(x, y)
            loss_d.append(d_loss)


            is_trainable(gen)
            not_trainable(dis)
            #g_loss = dis_on_gen.train_on_batch(ldr, [hdr, np.ones((bs,patch_size//4,patch_size//4,1))])
            g_loss = dis_on_gen.train_on_batch(ldr, [hdr, np.ones((bs,1))])
            loss_g.append(g_loss)
            
            print('epoch:%d, step:%d, d_loss:%f, g_loss[0]:%f, g_loss[1]:%f'%(epoch, step, d_loss, g_loss[0], g_loss[1]))
            step = step+1 

        gen.save_weights('./weights/'+save_path+'('+str(epoch)+')_gen.h5')
        dis.save_weights('./weights/'+save_path+'('+str(epoch)+')_dis.h5')
        loss_mean = np.mean(loss_d)
        loss_history_d.append(loss_mean)
        loss_mean = np.mean(loss_g)
        loss_history_g.append(loss_mean) 

    print ("epoch_total_time : %f"% (time.time()-epoch_star_time))
    print(loss_history_d)
    print(loss_history_g)

if __name__ == '__main__':
    save_path = '3_1_1'
    dis = discriminator_model(patch_size, patch_size, 4*ch)
    gen = generator_model(patch_size, patch_size, 3*ch)
    dis_on_gen = generator_containing_discriminator(patch_size, patch_size, 3*ch, gen, dis)    
    op = keras.optimizers.Adam(learning_rate=0.001)
    
    gen.compile(optimizer=op, loss='mse')
    #gen.load_weights('generator.h5')
    dis_on_gen.compile(optimizer=op, loss=[vgg_loss, 'mse'])
    dis.compile(optimizer=op, loss='mse')
    #dis.load_weights('discriminator.h5')
    
    train(gen, dis, dis_on_gen, save_path)

