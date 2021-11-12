
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
from utils import build_model, load_tflite, predict_tflite

app=Flask(__name__,template_folder='templates', static_folder='C:\\Users\\HPvns\\Desktop\\colorme\\img')


app.config['UPLOAD_FOLDER'] = 'C:\\Users\\HPvns\\Desktop\\colorme\\img\\Upload'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

interpreter = load_tflite()


# def load_tflite():
#     # Load the TFLite model and allocate tensors.    
#     interpreter = tf.lite.Interpreter(model_path="model.tflite")    
#     interpreter.allocate_tensors()
#     return interpreter

# # build model and load weight
# def build_model(): 
#     vgg = tf.keras.applications.VGG16()
#     model = Sequential()
#     for idx, layer in enumerate(vgg.layers):
#         if idx<19:
#             model.add(layer)
#     for layer in model.layers:
#         layer.trainable = False
#     model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2D(32, (3, 3), activation='relu', padding = 'same'))
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2D(16, (3, 3), activation='relu', padding = 'same'))
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2D(2, (3, 3), activation='tanh', padding = 'same'))
#     model.load_weights("C:\\Users\\HPvns\\Desktop\\colorme\\weight_file.h5")
#     model.compile(optimizer='adam', loss = 'mse', metrics=['accuracy'],)
#     return model




# def predict_tflite(filename):
#     test = img_to_array(load_img(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
#     test = resize(test, (224,224), anti_aliasing=True)
#     test*= 1.0/255
#     lab = rgb2lab(test)
#     l = lab[:,:,0]
#     L = np.repeat(l[..., np.newaxis], 3, -1)
#     L = L.reshape((1,224,224,3))

#     input_data = np.array(L, dtype=np.float32)
#     # Get input and output tensors.    
#     input_details = interpreter.get_input_details()    
#     output_details = interpreter.get_output_details()
#     #Predict model with processed data             
#     input_shape = input_details[0]['shape']
#     interpreter.set_tensor(input_details[0]['index'], input_data) 
#     print("invoking model")           
#     interpreter.invoke()     
#     print("invoking model Done")                  
#     ab = interpreter.get_tensor(output_details[0]['index']) 
#     ab = ab*128
#     cur = np.zeros((224, 224, 3))
#     cur[:,:,0] = l
#     cur[:,:,1:] = ab
#     imsave("img//output//out.jpg", lab2rgb(cur))




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
        # model = build_model()
        predict_tflite(filename, app, interpreter)
        return render_template('index.html', filename = filename)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(debug=True)


