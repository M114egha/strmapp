#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model


# Set background image
st.markdown(
    """
    <style>
        body {
            background-image: url('file:///Downloads/leafbackgrnd.jpg');  # Use the absolute path
            background-size: cover;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hide file uploader warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

# Get the current working directory
#current_directory = os.getcwd()

# Get the full path to the model file
# Use the absolute path to the model file
model_path = "C:/Users/HP/Downloads/best_model.h5"


# Load your pre-trained model
model = load_model(model_path)

# Assuming class_names is a list of disease names
class_names = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight"]



# Streamlit App Header
st.title("Plant Leaf Disease Prediction App")

# File Upload
uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Button to make prediction
    if st.button("Make Prediction"):
        try:
            # Load and preprocess the image for prediction
            img = load_img(uploaded_file, target_size=(256, 256))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img = np.expand_dims(img_array, axis=0)

            # Make prediction using your model
            prediction = model.predict(img)
            
            # Get the index of the predicted class
            predicted_class_index = np.argmax(prediction)

             # Display the predicted disease and its probability
            st.write(f"Predicted Disease: {class_names[predicted_class_index]} (Probability: {prediction[0][predicted_class_index]*100:.2f}%)")
        except Exception as e:
            st.write("Error making prediction:", str(e))


# In[ ]:





# In[ ]:




