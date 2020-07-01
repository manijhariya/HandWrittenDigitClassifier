# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 22:33:01 2019

@author: ExpertHub Intern Team-4,Hyderabad Dec,2019 
"""

# make a prediction for a new image.
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import PIL.Image


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

# load an image and predict the class
def run_example():
    # load the image
    y = "D:/ML & AI/Final project/static/sample1.png"
    
    img = load_image(y)
    # load model
    model = load_model('final_model.h5')
    # predict the class
    digit = model.predict_classes(img)
    print(digit[0])
    return (digit)

# entry point, run the example
run_example()
