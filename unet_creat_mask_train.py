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
import cv2
TRAIN_DIR = '/cell_image'
save_path='/wenjianhuizong/'

IMG_TYPE = '.png'         # Image type
IMG_CHANNELS = 3          # Default number of channels
IMG_DIR_NAME = 'images'   # Folder name including the image
MASK_DIR_NAME = 'masks'   # Folder name including the masks

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore', category=FutureWarning, module='skimage')

from scipy import ndimage as ndi
from skimage.morphology import erosion, square
imgs=[]
files=[]
masks=[]
def get_more_images(imgs):
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
    
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
    
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
    
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
    
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    
    more_images = np.concatenate((imgs,v,h))
    
    return more_images
def duplicate_labels(labels):
    more_images = []
    vert_flip_imgs = labels
    hori_flip_imgs = labels
    duplicate_labels = np.concatenate((labels,vert_flip_imgs,hori_flip_imgs))
    return duplicate_labels
def get_more_masks(masks):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
    for i in range(0,masks.shape[0]):
        av=cv2.flip(mask,1)
        ah=cv2.flip(mask,0)
        av=np.reshape(av,(256,256,1))
        ah=np.reshape(ah,(256,256,1))
        vert_flip_imgs.append(av)
        hori_flip_imgs.append(ah)
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)    
    more_masks=np.concatenate((masks,v,h))
    return more_masks
for dirpath,dirnames,filenames in os.walk(TRAIN_DIR): 
    for special_file in filenames:
        if 'crop' in special_file:
            files.append(special_file)
            img=imread(TRAIN_DIR+special_file)
            imgs.append(img)
            mask_file=special_file.replace('crop','labels')
            mask=imread(TRAIN_DIR+mask_file)
            mask = 100.0 * (mask[:,:,2] > 0)
            mask = ndimage.gaussian_filter(mask, sigma=(1.5, 1.5), order=0)
            
            print(np.shape(mask))
            if np.max(mask)!=0:
               mask=mask/np.max(mask)

            
            print(np.max(mask))
            mask=np.reshape(mask,(256,256,1))
            masks.append(mask)
print(len(masks))
print(len(imgs))
permutation = np.random.permutation(1550)
imgs=np.array(imgs)
masks=np.array(masks)
files=np.array(files)
imgs=imgs[permutation,:,:,:]
masks=masks[permutation]
files=files[permutation]
def read_image(observation_id, directory):
    return imread(directory+'/'+observation_id+'/images/'+observation_id+'.png')

def read_masks(observation_id, directory):
    str=directory+'/'+observation_id+'/masks'+ '/*.png'
    return io.ImageCollection(str)
def segment_mask(masks):
    '''Combine a list of masks into a single image.'''
    mask = np.sum(masks, axis=0)
    return np.clip(mask, 0, 1).astype(np.uint8)
def segment_soft_mask(masks):
    '''
    EXPERIMENTAL
    Try a soft encoding for masks (as opposed to a 0/1 hard encoding) where the probability of a 
    mask is a function of the distance from the center of the nearest nucleus.
    
    RESULT
    This didn't end up working out too well, the masks were way too small and required tuning of
    the cutoff parameter. 
    '''
    final_mask = np.zeros(masks[0].shape) # pixel locations with a value of 0 denote the background
    for i, mask in enumerate(masks):
        distance = ndi.distance_transform_edt(mask)
        final_mask = np.maximum(final_mask, distance)
    return final_mask / np.max(final_mask)
def segment_eroded_mask(masks, size=2):
    '''Remove pixels at the boundary of a mask. Useful for ensuring that no two masks are touching.'''
    masks = [erosion(mask, square(size)) for mask in masks]
    mask = np.sum(masks, axis=0)
    return np.clip(mask, 0, 1).astype(np.uint8)

def instance_mask(masks):
    '''Returns an overlay where each instance location is labeled by an integer starting at 1 and incresasing.'''
    all_labels = np.zeros(masks[0].shape) # pixel locations with a value of 0 denote the background
    for i, mask in enumerate(masks):
        mask = mask > 0
        label = (mask)*(i+1) # pixel locations with a value of i denote the ith mask
        all_labels = np.maximum(all_labels, label) # for overlapping masks, use the higher value - this shouldn't ever happen for this dataset
    return all_labels.astype(np.uint8)
def separate_instances(label_image):
    '''
    Input: Labeled pixel map where each integer corresponds with one nucleus. 
    Returns: A list of masks where each mask shows the complete pixel mapping for one nucleus.
    '''
    all_masks = []
    for i in range(1, np.max(label_image)+1):
        mask = (label_image == i).astype(np.uint8)
        all_masks.append(mask)
    return all_masks
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from skimage.morphology import label
from skimage.color import label2rgb

def encode_target(masks, w0=5, sigma=2):
    # ref : https://www.kaggle.com/piotrczapla/tensorflow-u-net-starter-lb-0-34/notebook

    weight = np.zeros(masks.shape)
    # calculate weight for important pixels
    distances = np.array([ndi.distance_transform_edt(m==0) for m in masks])
    shortest_dist = np.sort(distances, axis=0)
    # distance to the border of the nearest cell 
    d1 = shortest_dist[0]
    # distance to the border of the second nearest cell
    d2 = shortest_dist[1] if len(shortest_dist) > 1 else np.zeros(d1.shape)

    weight = w0 * np.exp(-(d1+d2)**2/(2*sigma**2)).astype(np.float32)
    weight = 1 + (merged_mask == 0) * weight
    return merged_mask - weight

def decode_target(encoding):
    target_mask = np.array(encoding == 0, dtype=np.uint8)
    weights = (-1 * encoding) + target_mask
    
    return target_mask, weights




from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy

import keras.backend as K

def weighted_binary_crossentropy(y_true, y_pred):
    '''
    Calculates the weighted pixel-wise binary cross entropy. Expects target to be encoded as `(mask - weights)`. 
    '''
    # mask <- where value==0
    target_mask = K.cast(K.equal(y_true, 0), 'float32') 
    
    # weights calculated as described above
    weights = (-1 * y_true) + target_mask
    
    cce = binary_crossentropy(target_mask, y_pred)  
    wcce = cce * K.squeeze(weights, axis=-1)
    return K.mean(wcce, axis=-1)
def prepare_target(observation):
    masks = read_masks(observation, TRAIN_DIR)
    masks_resized = [cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA) for mask in masks]
    encoding = encode_target(masks_resized)
    return encoding
def dice(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
def dice_loss(y_true,y_pred):
    return 1-dice(y_true,y_pred)
def softmax_dice_loss(y_true,y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.6 + dice_loss(y_true, y_pred) * 0.4

import cv2

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
from skimage import io,data
# images

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

def rgb_clahe_justl_ronghe(in_rgb_img):
    y=[]
    for x in in_rgb_img:
        bgr = x[:,:,[2,1,0]] # flip r and b
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l= clahe.apply(lab[:,:,0])
        a=np.mean(l)
        if a>127:
           l=255-l
        y.append(l)
    return y
def rgb_clahe(in_rgb_img):
    y=[]
    for x in in_rgb_img:
        bgr = x[:,:,[2,1,0]] # flip r and b
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb=bgr[:,:,[2,1,0]]
        y.append(rgb)
    return y

#imgs=rgb_clahe_justl(imgs)

imgs=np.array(imgs)
masks=np.array(masks)
print(np.shape(masks))
x_train=imgs[:-100,:,:,:]
x_val=imgs[-100:,:,:,:]
y_train=masks[:-100]
y_val=masks[-100:]
file_val=files[-100:]





from keras.preprocessing.image import ImageDataGenerator

data_gen_args = dict(featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.3,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.3,  # randomly shift images vertically (fraction of total height)
        zoom_range = 0.3,
        shear_range = 0.,
        horizontal_flip = True,  # randomly flip images
        vertical_flip = True, # randomly flip images
        fill_mode = 'constant'
        )  


image_datagen = ImageDataGenerator(rescale=1./255, **data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)


# Provide the same seed and keyword arguments to the flow methods
seed = 1
batch_size = 16
# ------ training data ------
train_image_generator = image_datagen.flow(x_train, batch_size=batch_size, seed=seed)
train_mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(train_image_generator, train_mask_generator)
train_steps = np.ceil(len(x_train) / batch_size)

# ------ validation data ------
val_image_generator = image_datagen.flow(x_val, batch_size=batch_size, seed=seed)
val_mask_generator = mask_datagen.flow(y_val, batch_size=batch_size, seed=seed)

# combine generators into one which yields image and masks
val_generator = zip(val_image_generator, val_mask_generator)
val_steps = np.ceil(len(x_val) / batch_size)
# define the u-net
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
def conv_bn_relu(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99,
                 bn_scale=True, use_bias=True):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x
def decoder_block_no_bn(input, filters, skip, block_name, activation='relu'):
    x = UpSampling2D()(input)
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1', activation=activation)
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2', activation=activation)
    return x
def conv_relu(input, num_channel, kernel_size, stride, name, padding='same', use_bias=True, activation='relu'):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = Activation(activation, name=name + '_relu')(x)
    return x
def create_pyramid_features(C1, C2, C3, C4, C5, feature_size=256):
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5', kernel_initializer="he_normal")(C5)
    P5_upsampled = UpSampling2D(name='P5_upsampled')(P5)

    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced',
                kernel_initializer="he_normal")(C4)
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4', kernel_initializer="he_normal")(P4)
    P4_upsampled = UpSampling2D(name='P4_upsampled')(P4)

    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced',
                kernel_initializer="he_normal")(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3', kernel_initializer="he_normal")(P3)
    P3_upsampled = UpSampling2D(name='P3_upsampled')(P3)

    P2 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced',
                kernel_initializer="he_normal")(C2)
    P2 = Add(name='P2_merged')([P3_upsampled, P2])
    P2 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2', kernel_initializer="he_normal")(P2)
    P2_upsampled = UpSampling2D(size=(2, 2), name='P2_upsampled')(P2)

    P1 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced',
                kernel_initializer="he_normal")(C1)
    P1 = Add(name='P1_merged')([P2_upsampled, P1])
    P1 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P1', kernel_initializer="he_normal")(P1)

    return P1, P2, P3, P4, P5

def prediction_fpn_block(x, name, upsample=None):
    x = conv_relu(x, 128, 3, stride=1, name="predcition_" + name + "_1")
    x = conv_relu(x, 128, 3, stride=1, name="prediction_" + name + "_2")
    if upsample:
        x = UpSampling2D(upsample)(x)
    return x





class SpeckleNoise(Layer):
    """Apply multiplicative one-centered Gaussian noise.
    This is useful to mitigate overfitting
    (you could see it as a form of random data augmentation).
    Speckle Noise (GS) is a natural choice as corruption process
    for real valued inputs.
    As it is a regularization layer, it is only active at training time.
    # Arguments
        stddev: float, standard deviation of the noise distribution.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

#    @interfaces.legacy_specklenoise_support
    def __init__(self, stddev, **kwargs):
        super(SpeckleNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs, training=None):
        def noised():
            return K.clip(inputs * K.random_normal(shape=K.shape(inputs),
                                            mean=1.,
                                            stddev=self.stddev), 0.0, 1.0)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev}
        base_config = super(SpeckleNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
def dice(y_true,y_pred,smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
def dice_loss(y_true,y_pred):
    return 1-dice(y_true,y_pred)
def softmax_dice_loss(y_true,y_pred):
    return binary_crossentropy(y_true, y_pred) * 0.7 + dice_loss(y_true, y_pred) * 0.3

def conv_block(inputs, filters, filter_size=3, drop_prob=0.2):
    x = Conv2D(filters, filter_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, filter_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SpatialDropout2D(drop_prob)(x)
    return x

def downsample(block):
    x = MaxPooling2D(pool_size=(2, 2)) (block)
    return x

def upsample(block, skip_connection, filters):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(block)
    stack = concatenate([skip_connection, x])
    return stack
def hard_dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
from keras.losses import binary_crossentropy
import os
import warnings

from keras.applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from xception_padding import Xception

def build_unet(IMG_WIDTH=256, IMG_HEIGHT=256, IMG_CHANNELS=6, drop_prob=0.2):
    
   # regularizer=regularizers.l2(0.00013)
    
    # ---- Model ----
    #inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
  
    
    # s = SpeckleNoise(0.005)(inputs)  #skimage speckel var defaults to 0.01
    # Downsample
    
    xception = Xception(input_shape=(256,256,6), weights='imagenet',include_top=False)
    conv1 = xception.get_layer("block1_conv2_act").output
    conv2 = xception.get_layer("block3_sepconv2_bn").output
    conv3 = xception.get_layer("block4_sepconv2_bn").output
    conv3 = Activation("relu")(conv3)
    conv4 = xception.get_layer("block13_sepconv2_bn").output
    conv4 = Activation("relu")(conv4)
    conv5 = xception.get_layer("block14_sepconv2_act").output

    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    outputs = Conv2D(1, (1, 1), activation='sigmoid', name="mask")(x)
    #model = load_model('unet_test_xception.h5', custom_objects={'weighted_binary_crossentropy': weighted_binary_crossentropy})
    #model.load_weights('unet_test_xception.h5') 
    model = Model(inputs=[model.input], outputs=[model.output])
    model.compile(optimizer='adam', loss=softmax_dice_loss)
    return model





model = build_unet()
model.summary()
from keras.callbacks import LearningRateScheduler

def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)

lr_sched = step_decay_schedule(initial_lr=0.001, decay_factor=0.85, step_size=10)
checkpointer = ModelCheckpoint('unet_1_nolab.h5',verbose=1, save_best_only=True)

results = model.fit_generator(train_generator, steps_per_epoch=train_steps, epochs=100, 
                              validation_data=val_generator, validation_steps=val_steps,
                              callbacks=[checkpointer, lr_sched])
model.load_weights('unet_1_nolab.h5')

imgs_1=[]
for img in x_val:
    img=img/255
    imgs_1.append(img)
imgs=imgs_1
imgs=np.array(imgs)
print(np.shape(imgs))
tests=model.predict(imgs)
pred=[]
print(np.shape(tests))
for test in tests:
    test=np.reshape(test,(256,256))
    pred.append(test)
for img,fil in zip(pred,files):
    imsave(save_path+fil,img)

plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

