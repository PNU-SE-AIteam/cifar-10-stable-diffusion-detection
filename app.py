import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('BESTcifakeCNN20240309-132543.keras')

def preprocess_image(uploaded_file):

    img = image.load_img(uploaded_file, target_size=(32, 32))

    img_array = image.img_to_array(img)
    # if alpha channel, ignore it
    if img_array.shape[-1] == 4: 
        img_array = img_array[:, :, :3] 
    img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title('Cifar 10 Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    if uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png":
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
        

        probability = prediction[0][0]
        st.write(f"The model predicts the probability of the image being real: {probability:.3f}")
    else:
        st.error("Please upload a JPEG or PNG image.")