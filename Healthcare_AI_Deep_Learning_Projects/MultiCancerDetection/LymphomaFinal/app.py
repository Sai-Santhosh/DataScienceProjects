
import os
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
model = load_model('ensemble.h5')
def preprocess_image(image):
    image = image.resize((224, 224)) 
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  
    return image
st.title('Lymphoma Classification using Ensemble of CNN Models')
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    if st.button('Predict'):
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        class_names = ["Chronic Lymphocytic Leukemia", "Follicular Lymphoma", "Mantle Cell Lymphoma"]  
        pred_class = class_names[np.argmax(prediction)]
        st.write(f"The image given has {pred_class}")
