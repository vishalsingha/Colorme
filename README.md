# Colorme [[see website]](http://vipul02vns.pythonanywhere.com/)



Colorme is a image colorization project which takes the black and white images as input and return the colored image as the output of the model.In this project we have taken taken 100K colored images and built a model using pretrained VGG16 model as a base model for feature extracture and than added few set of Conv2D and UpSampling 2D layer to get the output image.




# Dataset Source:<br>

The Dataset used in the training of model is taken from Kaggle from  two differnt places. LAter they have been merged to train the model.<br>
Dataset1 : [https://www.kaggle.com/darthgera/colorization](https://www.kaggle.com/darthgera/colorization)<br>
Dataset2 : [https://www.kaggle.com/yehonatan930/flickr-image-dataset-30k](https://www.kaggle.com/yehonatan930/flickr-image-dataset-30k)<br>

All the colored images were converted from RGB to LAB color channel. Further L channel is used as input to the model for predicting the AB color channel.
Since the VGG16 takes 3 channel image as input, so the L channel have been copied three times to make input as three channel and fed to te network for the predicting of AB channels.


# Model Artichecture: <br>

Model artichectre is made by adding pretrained VGG16 layers(up to 19 layers) and after that I have added Conv2D and UpSampling2D layer alternatively.<br><br> 
The Yellow layers are the pretrained VGG-16 layers.<br>
The gray layers are Upsampling 2D layer<br>
The skyble layers are Conv2D layer.<br><br>
<img src='https://github.com/vishalsingha/Colorme/blob/main/model_artichecture_block.png?raw=true' height= 600px weidth = 900px >


# Results: <br>
<pre>          <b>                                      Input grayscale Image                                       </b></pre>
<p float="left">
<img src='https://github.com/vishalsingha/Colorme/blob/main/input/input1.jpg?raw=true' height= 170px weidth = 250px >
<img src='https://github.com/vishalsingha/Colorme/blob/main/input/input3.jpg?raw=true'  height= 170px weidth = 250px>
<img src='https://github.com/vishalsingha/Colorme/blob/main/input/input4.jpg?raw=truee'  height= 170px weidth = 250px>
<img src='https://github.com/vishalsingha/Colorme/blob/main/input/input5.jpg?raw=truee'  height= 170px weidth = 250px>
<img src='https://github.com/vishalsingha/Colorme/blob/main/input/input6.jpg?raw=truee'  height= 170px weidth = 250px>
</p>


<pre>       <b>                                        Output Colored Images                                       </b></pre>
<p float="left">
<img src='https://github.com/vishalsingha/Colorme/blob/main/output/res1.jpg?raw=truee'  height= 170px weidth = 250px>
<img src='https://github.com/vishalsingha/Colorme/blob/main/output/res3.jpg?raw=truee'  height= 170px weidth = 250px>
<img src='https://github.com/vishalsingha/Colorme/blob/main/output/res4.jpg?raw=truee'  height= 170px weidth = 250px>
<img src='https://github.com/vishalsingha/Colorme/blob/main/output/res5.jpg?raw=truee'  height= 170px weidth = 250px>
<img src='https://github.com/vishalsingha/Colorme/blob/main/output/res6.jpg?raw=truee'  height= 170px weidth = 250px>

</p>

# Deployment:
Colorme webapp for the demo purpose is deployed via flask framework. For increasing the speed of the model regular tensorflow model was converted into <b>Tensorflow Lite</b> which increase the latency of the model much.

```python
# convert to tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

```

Deployment link: [http://vipul02vns.pythonanywhere.com/](http://vipul02vns.pythonanywhere.com/)

# To Do list
- [x] Model Training
- [x] Deployment through flask
- [x] Conversion regular tensoflow to tflite and deploy
- [ ] Deployment via docker
