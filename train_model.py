#import library
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

# loadinng pre-trained vgg16 model
vgg = tf.keras.applications.VGG16()


model = Sequential()

# adding layers of vgg16 to our model
for idx, layer in enumerate(vgg.layers):
  if idx<19:
    model.add(layer)

# freezing the layers of vgg16 to non trainable
for layer in model.layers:
  layer.trainable = False

# adding some custom layers
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

# compile the model
model.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'],)

# train_data path
COLOR_PATH = '/content/train_col_images'
# validation data path
VAL_PATH = '/content/validation_col_images'

# image data geneator
train_datagen = ImageDataGenerator(rescale=1. / 255)

# getting the batch from train_data folder
train = train_datagen.flow_from_directory(COLOR_PATH, target_size=(224, 224), batch_size=64, class_mode=None)
# getting the batch from the validation_data folder
val = train_datagen.flow_from_directory(VAL_PATH, target_size = (224, 224), batch_size = 64, class_mode = None)

EPOCHS = 30


TRAIN_LOSS = []    #for storing training loss for each epoch
VAL_LOSS = []      #for storing training loss for each epoch

#--------------------------------------------------Training loop-------------------------------------------#
for epoch in range(EPOCHS):
  
  train_loss = 0
  val_loss = 0
  
  #loop for the model training
  for i in tqdm_notebook(range(len(train))):
    batch = train[i]
    #convert rgb image to lab
    batch_lab = rgb2lab(batch)  
    #copy l channel to all three channel for inputting to model
    x_batch = np.repeat(batch_lab[:, :, :, 0][..., np.newaxis], 3, -1)
    #extract a and b channel as true prediction
    y_batch = batch_lab[:, :, :, 1:]/128
    #fit the model on batch
    h = model.fit(x_batch, y_batch, verbose=0)
    #calculate loss
    loss = h.history['loss']
    train_loss = train_loss + loss[0]

  #loop for model validation
  for j in range(len(val)):
    batch_lab = val[j]
    #convert rgb image to lab
    batch_lab = rgb2lab(batch)
    #copy l channel to all three channel for inputting to model
    x_batch = np.repeat(batch_lab[:, :, :, 0][..., np.newaxis], 3, -1)
    #extract a and b channel as true prediction
    y_batch = batch_lab[:, :, :, 1:]/128
    #evaluate on validation set
    h = model.evaluate(x_batch, y_batch, verbose = 0) 
    #calculate loss
    val_loss = val_loss + h[0]

  #print the train and validation loss for each epoch
  print(f"## EPOCH:{epoch+1} Completed!! train_loss: {train_loss/len(train)}  || val_loss : {val_loss/len(val)}")
  
  #append the train and validation loss for each epoch
  TRAIN_LOSS.append(train_loss)
  VAL_LOSS.append(val_loss)
  
# save the model weight
model.save_weights('/content/drive/MyDrive/data/image_colorization-v3.h5')

