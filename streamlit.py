import streamlit as st
import tensorflow as tf 
from tensorflow.keras.preprocessing import image 
import numpy as np
from PIL import Image
import cv2
import os

#loading the model 
fruit_model= tf.keras.models.load_model('fruit_quality_model_final1.h5')
rotten_model=tf.keras.models.load_model('unet_mobilenetv201.h5')



fruit_labels={
   0:'fresh',
    1:'rotten',
}

#preprocessing the image 
def preprocess_image(img):
    img = img.resize((180, 180))    
    img = np.array(img)
    img=img/255
    img = np.expand_dims(img, axis=0)
    return img

def rotten_percentage(pil_image):

    image = pil_image.resize((224, 224))  
    image_np = np.array(image) / 255.0
    image_np = np.expand_dims(image_np, axis=0)

    # Predict mask
    prediction = rotten_model.predict(image_np)[0]
    mask = np.argmax(prediction, axis=-1)  

    # Count pixels
    rotten_pixels = np.sum(mask == 1)
    fruit_pixels = np.sum(mask == 2)

    total_fruit_area = rotten_pixels + fruit_pixels
    if total_fruit_area == 0:
        return 0.0  

    rotten_percent = (rotten_pixels / total_fruit_area) * 100
    return round(rotten_percent, 2)


#streamlit 

st.title('Fruit_Rotten_Fresh_Detection')
st.write('UPLOAD THE IMAGE OF THE FRUIT ')
uploaded_image= st.file_uploader("upload the image", type=['jpg', 'png', 'jpeg'])

if uploaded_image is not None:
    img= Image.open(uploaded_image)
    new_img=img.copy()
    st.image(img, caption='Uploaded Image', use_column_width=True) 

    img= preprocess_image(img)
    with st.spinner('Processing...'):
        predictions= fruit_model.predict(img)
        true_label=np.argmax(predictions)
        result=fruit_labels[int(true_label)]
        if result == 'rotten':
            percentage=rotten_percentage(new_img)
            st.write(f'The fruit is {result} with a rotten percentage of {percentage}%')
        else:
            st.success(f'the fruit is {result}')



