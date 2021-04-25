

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


app=Flask(__name__,template_folder='templates')


app.config['UPLOAD_FOLDER'] = 'C:\\Users\\Vipul Singh\\Desktop\\colorme\\Upload'



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
    model.load_weights("C:\\Users\\Vipul Singh\\Desktop\\colorme\\weight_file.h5")
    model.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'],)
    return model


def predict(filename,  model):
    test = img_to_array(load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
    test = resize(test, (224,224), anti_aliasing=True)
    test*= 1.0/255
    lab = rgb2lab(test)
    l = lab[:,:,0]
    L = np.repeat(l[..., np.newaxis], 3, -1)
    L = L.reshape((1,224,224,3))
    ab = model.predict(L)
    #print(ab.shape)
    ab = ab*128
    cur = np.zeros((224, 224, 3))
    cur[:,:,0] = l
    cur[:,:,1:] = ab
    imsave("output//o2.jpg", lab2rgb(cur))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        file = request.files['upload']
        if file.filename=="":
            return render_template('index.html', text = 'Please choose a file to upload')
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        model = build_model()
        predict(filename,  model)
        return render_template('index.html', filename = filename)


if __name__ == '__main__':
    app.run(debug=True)


