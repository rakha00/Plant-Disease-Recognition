import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Set page title and description
st.set_page_config(page_title="Plant Disease Recognition", layout="wide")
st.title("Plant Disease Recognition - BOCIL NI BOS")
st.write("Upload a leaf image to detect if it's healthy or has a disease.")

# Load the trained model
@st.cache_resource
def load_model():
    return keras.models.load_model('plant_disease_recognition_cnn_model.keras')

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Define class labels
class_labels = ['Healthy', 'Powdery', 'Rust']

# Create file uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image_bytes = uploaded_file.getvalue()
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, width=300)
    
    # Process the image and make prediction
    if model_loaded:
        with col2:
            st.subheader("Prediction Result")
            with st.spinner("Analyzing image..."):
                # Preprocess the image
                img = img.resize((224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize
                
                # Make prediction
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]
                predicted_label = class_labels[predicted_class]
                confidence = float(predictions[0][predicted_class]) * 100
                
                # Display result with appropriate styling
                if predicted_label == "Healthy":
                    st.success(f"Prediction: {predicted_label}")
                    st.balloons()
                else:
                    st.warning(f"Prediction: {predicted_label} Disease")
                
                st.write(f"Confidence: {confidence:.2f}%")
                
                # Display all probabilities
                st.write("Probability distribution:")
                for i, label in enumerate(class_labels):
                    prob = float(predictions[0][i]) * 100
                    st.write(f"{label}: {prob:.2f}%")
                    st.progress(prob/100)

# Add information about the diseases
with st.expander("About Plant Diseases"):
    st.write("""
    ### Plant Diseases Information:
    
    - **Healthy**: The leaf shows no signs of disease and appears normal.
    
    - **Powdery Mildew**: A fungal disease that appears as white powdery spots on the leaves. 
    It can reduce photosynthesis and plant vigor.
    
    - **Rust**: A fungal disease characterized by rusty-colored spots on leaves. 
    Severe infections can cause premature leaf drop and reduced plant growth.
    """)

# Add footer
st.markdown("---")
st.caption("Plant Disease Recognition App | Created with Streamlit and TensorFlow")