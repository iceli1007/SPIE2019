import warnings
import os
from pathlib import Path
import skimage.io as io
import numpy as np
import pandas as pd
import skimage
import glob
from skimage.io import imread, imshow, imread_collection, concatenate_images,imsave
import matplotlib.pyplot as plt
from keras.losses import categorical_crossentropy
import scipy.ndimage as ndimage
from skimage import transform,exposure
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy

import keras.backend as K
import cv2
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation,UpSampling2D, Conv2D, BatchNormalization, Activation, concatenate, Add
from keras.layers.core import Lambda, SpatialDropout2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras import metrics
import tensorflow as tf
from keras.utils import get_file
from  keras.engine import Layer
import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
train_image_path='/data5/image/breastpathq/datasets/test/'
save_path='/data5/badcode/_temp/cell_counting_v2/lzq_train/test_mask/'
from scipy import ndimage as ndi
from skimage.morphology import erosion, square
imgs=[]
mask_files=[]

for dirpath,dirnames,filenames in os.walk(train_image_path): 
    for special_file in filenames:
        mask_file=special_file.replace('.tif','_mask')
        mask_file=mask_file+'.tif'
        mask_files.append(mask_file)
        img=imread(train_image_path+special_file)
        #img=transform.resize(img, (256, 256,3),mode='reflect')
        imgs.append(img)
imgs=np.array(imgs)
print(np.shape(imgs))
def dice(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
def dice_loss(y_true,y_pred):
    return 1-dice(y_true,y_pred)
def softmax_dice_loss(y_true,y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.6 + dice_loss(y_true, y_pred) * 0.4
def rgb_clahe_justl(in_rgb_img):
    y=[]
    for x in in_rgb_img:
        bgr = x[:,:,[2,1,0]] # flip r and b
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l= clahe.apply(lab[:,:,0])
        a=np.mean(l)
        if a>127:
           l=255-l
        lab[:,:,0]=l
        y.append(lab)
    return y

imgs=rgb_clahe_justl(imgs)

batch_size=1
test_steps = np.ceil(len(imgs) / batch_size)
imgs_1=[]
for img in imgs:
    img=transform.resize(img, (256, 256,3),mode='reflect')
    print(np.max(img))
    img=img*255
    #img=img.astype(np.int32)
    print(np.max(img))
    imgs_1.append(img)
imgs=imgs_1
model = load_model('creat_mask.h5', custom_objects={'softmax_dice_loss': softmax_dice_loss})
model.load_weights('creat_mask.h5')
#tests = model.predict_generator(IMG,steps=test_steps)
imgs_1=[]
for img in imgs:
    img=img/255
    imgs_1.append(img)
imgs=imgs_1
imgs=np.array(imgs)
print(np.shape(imgs))
tests=model.predict(imgs)
pred=[]
print(np.shape(tests))
for fil,test in zip(mask_files,tests):
    print(fil)
    test=np.reshape(test,(256,256))
    #imshow(test)
    #plt.show()
    print(np.max(test))
    pred.append(test)
for img,fil in zip(pred,mask_files):
    #img=np.reshape(img,(256,256,1))
    #print(np.shape(img))
    img=img*255
    img=img.astype(np.uint8)
    img=np.reshape(img,(256,256))
    imsave(save_path+fil,img)

