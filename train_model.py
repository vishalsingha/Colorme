import os
import numpy as np
import keras
import tensorflow as tf
import tqdm
from tqdm import tqdm_notebook
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.transform import resize
from skimage.io import imsave, imread


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

model.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'],)


COLOR_PATH = '/content/train_col_images'
VAL_PATH = '/content/validation_col_images'


train_datagen = ImageDataGenerator(rescale=1. / 255)


train = train_datagen.flow_from_directory(COLOR_PATH, target_size=(224, 224), batch_size=64, class_mode=None)
val = train_datagen.flow_from_directory(VAL_PATH, target_size = (224, 224), batch_size = 64, class_mode = None)

EPOCHS = 30

TRAIN_LOSS = []
VAL_LOSS = []

for epoch in range(EPOCHS):
  train_loss = 0
  val_loss = 0
  for i in tqdm_notebook(range(len(train))):
    batch = train[i]
    batch_lab = rgb2lab(batch)
    x_batch = np.repeat(batch_lab[:, :, :, 0][..., np.newaxis], 3, -1)
    y_batch = batch_lab[:, :, :, 1:]/128
    h = model.fit(x_batch, y_batch, verbose=0)
    loss = h.history['loss']
    train_loss = train_loss + loss[0]


  for j in range(len(val)):
    batch_lab = val[j]
    batch_lab = rgb2lab(batch)
    x_batch = np.repeat(batch_lab[:, :, :, 0][..., np.newaxis], 3, -1)
    y_batch = batch_lab[:, :, :, 1:]/128
    h = model.evaluate(x_batch, y_batch, verbose = 0) 
    val_loss = val_loss + h[0]

  
  print(f"## EPOCH:{epoch+1} Completed!! train_loss: {train_loss/len(train)}  || val_loss : {val_loss/len(val)}")
  TRAIN_LOSS.append(train_loss)
  VAL_LOSS.append(val_loss)
  
model.save_weights('/content/drive/MyDrive/data/image_colorization-v3.h5')

