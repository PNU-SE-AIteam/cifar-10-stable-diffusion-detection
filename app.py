import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os 
import shutil

# Load the model
model = load_model('BESTcifakeCNN20240411-184011.keras')

def preprocess_image_and_get_image(uploaded_file, crop_area):
    img = image.load_img(uploaded_file)
    width, height = img.size
    left, top, right, bottom = crop_area
    img = img.crop((left * width, top * height, right * width, bottom * height))
    img = img.resize((32, 32))
    img_array = image.img_to_array(img)
    # if alpha channel, ignore it
    if img_array.shape[-1] == 4: 
        img_array = img_array[:, :, :3] 
    img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img 

st.title('Cifar 10 Image Classifier')
st.write("This model is designed to distinguish between real images and AI-generated ones. It was trained on the CIFAKE dataset (60,000 fake and 60,000 real 32x32 RGB images collected from CIFAR-10)")
uploaded_files = st.file_uploader("Choose images to evaluate...", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png":
            # Add crop functionality
            left, top, right, bottom = st.sidebar.slider("Crop Area", 0.0, 1.0, (0.0, 1.0), 0.05)
            img_array, img = preprocess_image_and_get_image(uploaded_file, (left, top, right, bottom))
            prediction = model.predict(img_array)
            
            probability = prediction[0][0]
            if probability > 0.5:
                st.write(f"The image below IS real.")
            else:
                st.write(f"The image below is AI-generated.")
            
            st.image(img, caption="32x32 Image", width=100)
        else:
            st.error("Please upload JPEG or PNG images.")
