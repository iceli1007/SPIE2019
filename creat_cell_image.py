import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
#from torch.utils.data import DataLoader, TensorDataset
#from torchvision.transforms import transforms
from skimage import transform,exposure,img_as_ubyte
from skimage.io import imread, imshow, imread_collection, concatenate_images,imsave
import matplotlib.pyplot as plt
import skimage.io as io
import skimage
import pandas as pd
import csv
import keras.backend.tensorflow_backend as KTF
SAVE_PATH='/cell_image/'
path='/cells/images'
label_path='/cells/labels/'
image_files=[]
mask_files=[]
images=[]
masks=[]
IMG_HEIGHT=256
IMG_WIDTH=256
def seg_image(img,step_size=100,to_size=256,):

    a=img.shape[0]
    a_step=int(a/step_size)
    b=img.shape[1]
    b_step=int(b/step_size)
    c=img.shape[2]
    img_list=[]
    for i in range(a_step):
        if (i+1)*step_size+to_size>a:
            for j in range(b_step):
                if (j+1)*step_size+to_size>b:
                    img_1=img[i*step_size:a,j*step_size:b,:]
                    #print('4')
                    img_1=transform.resize(img_1, (to_size, to_size,c),mode='reflect')
                    #img_1=img_1*255
                    img_1=img_as_ubyte(img_1)
                    img_list.append(img_1)
                    break
                else:
                    img_1=img[i*step_size:a,j*step_size:j*step_size+to_size,:]
                    #print('3')
                    img_1=transform.resize(img_1, (to_size, to_size,c),mode='reflect')
                    #img_1=img_1*255
                    img_1=img_as_ubyte(img_1)
                    img_list.append(img_1)
            break
        else:
            for j in range(b_step):
                if (j+1)*step_size+to_size>b:
                    img_1=img[i*step_size:i*step_size+to_size,j*step_size:b,:]
                    img_1=transform.resize(img_1, (to_size, to_size,c),mode='reflect')
                    #img_1=img_1*255
                    img_1=img_as_ubyte(img_1)
                    img_list.append(img_1)
                    #print('2')
                    break
                else:
                    img_1=img[i*step_size:i*step_size+to_size,j*step_size:j*step_size+to_size,:]
                    img_list.append(img_1)
                    #print('1')
      
    #print(len(img_list))
    #print(a,b)
    #print(a_step,b_step)

    return img_list

for dirpath,dirnames,filenames in os.walk(label_path): 
    for special_file in filenames:
        special_file_1=special_file[:-4]
        print(special_file_1)
        img_file=special_file.replace('labels','crop')
        img_file_1=img_file[:-4]
        img=imread(path+img_file)
        seg_img=seg_image(img)
        print(img_file_1)
        for i,img in enumerate(seg_img):
            save_path=SAVE_PATH+img_file_1+'_'+str(i)+'.tif'
            #print(type(save_path))
            
            imsave(save_path,img)
            
        mask=imread(label_path+special_file)
        print(len(seg_img))
        seg_mask=seg_image(mask)
        for i,msk in enumerate(seg_mask):
            str_value=str(i)
            imsave(SAVE_PATH+special_file_1+'_'+str_value+'.tif',msk)
        print(len(seg_mask))

