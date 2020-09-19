import os
import re

import numpy as np

from PIL import Image


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow as tf


def model():
    input_img=Input(shape=(256,256,3))
    l1=Conv2D(64,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(input_img)
    l2=Conv2D(64,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(l1)
    l3=MaxPooling2D(padding='same')(l2)
    l4=Conv2D(128,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(l3)
    l5=Conv2D(128,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(l4)
    l6=MaxPooling2D(padding='same')(l5)
    l7=Conv2D(256,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(l6)
    l8=UpSampling2D()(l7)
    l9=Conv2D(128,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(l8)
    l10=Conv2D(128,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(l9)
    l11=add([l5,l10])
    l12=UpSampling2D()(l11)
    l13=Conv2D(64,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(l12)
    l14=Conv2D(64,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(l13)
    l15=add([l2,l14])
    decoder=Conv2D(3,(3,3),padding='same',activation='relu',activity_regularizer=regularizers.l1(10e-10))(l15)
    autoencoder=Model(input_img,decoder)
    #autoencoder.summary();

    opt = tf.keras.optimizers.Adam()

    autoencoder.compile(optimizer=opt,loss='mean_squared_error',metrics=['accuracy'])
    autoencoder.load_weights('static/weights/weights.h5')
    return  autoencoder


