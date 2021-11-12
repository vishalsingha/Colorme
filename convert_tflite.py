
from flask import Flask, request, redirect, url_for, render_template, flash
import numpy as np
import tensorflow as tf
import keras
import  os
from werkzeug.utils import secure_filename
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.preprocessing.image import  img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.transform import resize
from skimage.io import imsave, imread

# build model and load weight
def build_model(): 
    vgg = tf.keras.applications.VGG16()
    model = Sequential()
    for idx, layer in enumerate(vgg.layers):
        if idx<19:
            model.add(layer)
    for layer in model.layers:
        layer.trainable = False
    model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding = 'same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(2, (3, 3), activation='tanh', padding = 'same'))
    model.load_weights("C:\\Users\\HPvns\\Desktop\\colorme\\weight_file.h5")
    model.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'],)
    return model

model = build_model()


# convert to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the tflite model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)


