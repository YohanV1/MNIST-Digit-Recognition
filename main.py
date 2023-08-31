import tensorflow as tf
from keras.models import load_model
from PIL import Image as im
import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
import cv2

@st.cache_resource
def load_data():
    mnist_model = load_model('model.h5')
    return mnist_model


model = load_data()

st.header("Handwritten Digit Recognition")
st.write("using a neural network for multiclass classification with softmax")
canvas_result = st_canvas(
    fill_color="grayscale(0, 0, 0, 0.0)",
    stroke_width=15,
    stroke_color='white',
    background_color='black',
    height=280,
    width=280,
    drawing_mode='freedraw',
    key="canvas",
)

image = canvas_result.image_data # this is an array

data = im.fromarray(image) # this is an image

data = data.resize((28, 28)) # resized image

data = data.convert('L')

img_array = np.array(data)

img_array = img_array.reshape(1, 784)

prediction = model.predict(img_array)
prediction_sm = tf.nn.softmax(prediction)

st.write(f" Largest prediction index: {np.argmax(prediction_sm)}")
