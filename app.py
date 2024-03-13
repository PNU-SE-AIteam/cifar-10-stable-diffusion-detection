import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model
model = load_model('BESTcifakeCNN20240303.keras')

def preprocess_image(uploaded_file):
    # Load the image and resize it to 32x32
    img = image.load_img(uploaded_file, target_size=(32, 32))
    
    # Convert the image to an array
    img_array = image.img_to_array(img)
    # if alpha channel, ignore it
    if img_array.shape[-1] == 4: 
        img_array = img_array[:, :, :3] 
    img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

st.title('CIFAR-10 Image Classifier')
st.write("This model is designed to distinguish between real images and AI-generated ones. It was trained on the CIFAKE dataset (60,000 stable-diffusion-generated and 60,000 real 32x32 RGB images collected from CIFAR-10)")
uploaded_file = st.file_uploader("Choose an image to evaluate...", type=["jpg", "png"])

if uploaded_file is not None:
    if uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png":
        img_array = preprocess_image(uploaded_file)
        prediction = model.predict(img_array)
        
        probability = prediction[0][0]
        if probability > 0.5:
            st.write(f"The image IS real.")
        else:
            st.write(f"The image is AI-generated.")
        
        # Display the 32x32 version of the image
        st.image(uploaded_file, caption="32x32 Version of the Uploaded Image", use_column_width=True)
        
    else:
        st.error("Please upload a JPEG or PNG image.")