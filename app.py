import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('fine_tuned_model.h5')

# Class names
class_names = [
    'chicken_curry', 'hamburger', 'pizza', 'sushi', 'ice_cream', 
    'ramen', 'fried_rice', 'chicken_wings', 'grilled_salmon', 'steak'
]

def preprocess_image(img):
    resized_image = cv2.resize(image, (224,224))
    resized_image = np.asarray(resized_image)
    resized_image = resized_image / 255.
    resized_image = np.expand_dims(resized_image, axis=0)
    return resized_image


# Streamlit app
st.title('Food Image Classification')
st.write('Upload an image of food to classify it.')

# Sidebar with class names
st.sidebar.title("The model can predict the following classes:")
for class_name in class_names:
    st.sidebar.write(class_name)

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type='jpg')

if uploaded_file is not None:
    # Load the uploaded image
    img = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button("Process"):
            # Convert PIL image to OpenCV format
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Resize image to (224, 224)
            resized_image = preprocess_image(cv2_image)
            resized_image = tf.constant(resized_image)
            model_output = model.predict(resized_image)
            pred_class = class_names[np.argmax(model_output)]
            confidence = np.max(model_output)
    
    
    # Display the prediction
    st.write(f'Prediction: {pred_class}')
    st.write(f'Confidence: {confidence:.2f}')
