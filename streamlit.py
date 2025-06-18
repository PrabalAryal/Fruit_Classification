import streamlit as st
import tensorflow as tf 
import numpy as np
from PIL import Image

# Load models=
fruit_model = tf.keras.models.load_model('fruit_quality_model.h5')
rotten_model = tf.keras.models.load_model('unet_mobilenetv2.h5')

fruit_labels = {
    0: 'fresh',
    1: 'rotten',
}

# Preprocessing
def preprocess_image(img):
    img = img.resize((180, 180))    
    img = np.array(img)
    img = img[:, :, ::-1]  # Convert RGB to BGR
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def rotten_percentage(pil_image):
    image = pil_image.resize((224, 224))  
    image_np = np.array(image)/255.0
    image_np = np.expand_dims(image_np, axis=0)
    prediction = rotten_model.predict(image_np)[0]
    mask = np.argmax(prediction, axis=-1)

    rotten_pixels = np.sum(mask == 1)
    fruit_pixels = np.sum(mask == 2)

    total_fruit_area = rotten_pixels + fruit_pixels
    if total_fruit_area == 0:
        return 0.0

    rotten_percent = (rotten_pixels / total_fruit_area) * 100
    return round(rotten_percent, 2)

# Streamlit UI
st.title('🍎 Fruit Rotten/Fresh Detection')

# Choose input method
input_method = st.selectbox("Choose input method", ["Upload Image", "Use Camera"])

img = None


if input_method == "Upload Image":
    uploaded_image = st.file_uploader("Upload a fruit image", type=['jpg', 'jpeg', 'png'])
    if uploaded_image:
        img = Image.open(uploaded_image)

elif input_method == "Use Camera":
    camera_image = st.camera_input("Take a fruit picture")
    if camera_image:
        img = Image.open(camera_image)


if img:
    st.image(img, caption='Input Image', use_column_width=True)
    if st.button("Predict"):
        with st.spinner('Analyzing...'):
            new_img = img.copy()
            processed_img = preprocess_image(img)

            predictions = fruit_model.predict(processed_img)
            true_label = np.argmax(predictions)
            result = fruit_labels[int(true_label)]

            if result == 'rotten':
                percentage = rotten_percentage(new_img)
                st.error(f'The fruit is **{result}** with a rotten percentage of **{percentage}%**')
            else:
                st.success(f'The fruit is **{result}**')
