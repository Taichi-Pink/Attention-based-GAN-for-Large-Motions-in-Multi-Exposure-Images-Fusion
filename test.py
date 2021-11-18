import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import os
import matplotlib.pyplot as plt
import math, cv2, glob, time
import numpy as np
import imageio as io
from load_data import load_data
from model import *

##test config
w  = 1280
h  = 768
ch = 3
batch_size = 1


def test(model):
  psnr = []
  ssim = []

  data_dir   = '/home/ziyi.liu1/gan/gan/dataset/test/'
  scene_dirs = [scene_dir for scene_dir in os.listdir(data_dir) if scene_dir!="Label"]
  scene_dirs = sorted(scene_dirs)
  num_scenes = len(scene_dirs)
  print(num_scenes)
  
  t_total = 0.0
  for index in range(num_scenes):
  #for index in range(1):
    cur_path = scene_dirs[index]
    cur_path = os.path.join(data_dir, cur_path)
    
    over_exp  = cv2.imread(os.path.join(cur_path, 'input_3.tif'))
    over_exp  = over_exp[:, :, ::-1]
    med_exp   = cv2.imread(os.path.join(cur_path, 'input_2.tif'))
    med_exp   = med_exp[:, :, ::-1]
    under_exp = cv2.imread(os.path.join(cur_path, 'input_1.tif'))
    under_exp = under_exp[:, :, ::-1]
    
    ''' finding corresponding ldr image ''' 
    label_p = os.path.join(cur_path, 'ref_hdr_aligned.hdr')    
    label   = cv2.imread(label_p)
    label   = label[:, :, ::-1]
    
    over_exp  = cv2.resize(over_exp, (w, h))
    med_exp   = cv2.resize(med_exp, (w, h))
    under_exp = cv2.resize(under_exp, (w, h))
    label     = cv2.resize(label, (w, h))   
    
    ''' bring to [0,1] '''
    over_exp  = norm_0_to_1(over_exp)
    med_exp   = norm_0_to_1(med_exp)
    under_exp = norm_0_to_1(under_exp)
    label     = norm_0_to_1(label) 
    
    img = np.concatenate([under_exp, med_exp, over_exp], axis=2)
    img = np.expand_dims(img, axis=0)          
    img = img*2.0-1.0
    
    t_start = time.time()
    ldr     = model.predict(img)
    t_total = t_total + time.time() - t_start
    
    ldr     = np.squeeze(ldr)
    ldr_    = ((ldr+1.)/2.)   
    tem = ldr_*255.
    tem = tem.astype(np.uint8)
    
    #if no%10 ==0:
      #plt.imshow(tem)
      #plt.show()
    #io.imwrite('./results/'+model_name+'/' + path+ '_'+ str(no) + '.png', tem)

    psnr_result = tf.image.psnr(ldr_, label, max_val=1.0)
    ssim2       = tf.image.ssim(tf.squeeze(ldr_), tf.squeeze(label), max_val=1.0)
    #print('no%d, psnr_result:%f, ssim:%f'%(no, psnr_result, ssim2))
    #io.imwrite(results_path + str(no) + '.png', tem)
    ssim.append(ssim2)
    psnr.append(psnr_result)
    
  print('total_time:', t_total)
  print('average_psnr:', np.mean(psnr))
  print('average_ssim:', np.mean(ssim))


if __name__ == '__main__':
    gen = generator_model(h, w, 3*ch)
    print(gen.summary())
    op  = keras.optimizers.Adam(learning_rate=0.001)    
    gen.compile(optimizer=op, loss='mse')
    gen.load_weights('./weights/3_1_1(0)_gen.h5')
    test(gen)
    gen.load_weights('./weights/3_1_1(1)_gen.h5')
    test(gen)
    gen.load_weights('./weights/3_1_1(2)_gen.h5')
    test(gen)
    gen.load_weights('./weights/3_1_1(3)_gen.h5')
    test(gen)
    gen.load_weights('./weights/3_1_1(4)_gen.h5')
    test(gen)
