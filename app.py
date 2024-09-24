import joblib
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import base64

# Set page configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string

# Replace with your image path for background
image_path = '/Users/mba/Documents/Work/BrainTumor/BG2.jpeg'
base64_image = encode_image(image_path)

# Inject custom CSS to set the background image, blur it, and style elements
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-color: rgba(0, 0, 0, 0.8); /* Faded effect */
        color: white;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.5); /* Optional dark overlay */
        backdrop-filter: blur(8px); /* Adjust the blur amount here */
        z-index: -1; /* Send to back */
    }}
    .stHeader {{
        color: white;
        text-align: center;
        font-size: 2.5rem;
        font-weight: bold;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
    }}
    .prediction-box {{
        background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent box */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
        color: white;
        margin-top: 20px;
        font-size: 1.2rem; /* Increase the font size */
    }}
    .prediction-box h2 {{
        color: red;
        font-size: 2rem; /* Larger font size for heading */
    }}
    .prediction-box p {{
        font-size: 1.5rem; /* Larger font size for text */
    }}

    .image-box {{
        display: flex;
        justify-content: flex-end;  /* Align the image to the right */
        padding-left: 1000px;        /* Add some padding on the right */
    }}
    </style>
    """, unsafe_allow_html=True)

# Header with custom styling
st.markdown('<h1 class="stHeader">Brain Tumor Detection Model</h1>', unsafe_allow_html=True)

# Load the pre-trained model
svm_model = joblib.load('/Users/mba/Documents/Work/BrainTumor/svm_model.pkl')  # Update with your model path

# Define the categories for classification
data_cat = ['No Tumor', 'Pituitary Tumor']

# Define image dimensions
img_height = 200
img_width = 200

# Upload image file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # Load and preprocess the image
    image_load = Image.open(uploaded_file).convert("L").resize((img_width, img_height))  # Convert to grayscale
    img_arr = np.array(image_load)  # Convert image to array
    img_arr = img_arr.reshape((img_height, img_width, 1))  # Reshape to (200, 200, 1)
    img_bat = np.expand_dims(img_arr, axis=0)  # Add batch dimension
    img_bat = img_bat.reshape(1, -1) / 255  # Reshape for SVM and normalize the image

    # Make predictions
    prediction = svm_model.predict(img_bat)
    prediction_label = data_cat[prediction[0]]
    confidence_score = svm_model.decision_function(img_bat)
    confidence = np.max(confidence_score)

    # If the model detects a tumor, draw the red box
    if prediction[0] == 1:  # If a tumor is detected
        # Draw the red box on the original image (before grayscale conversion)
        img_with_box = np.array(Image.open(uploaded_file).resize((img_width, img_height)).convert("RGB"))  # Convert PIL image to RGB array
        
        # Example coordinates for the box (you can adjust based on tumor detection logic)
        x1, y1, x2, y2 = 50, 50, 150, 150  # Replace with the actual coordinates from the model's output
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw red box

        # Convert back to PIL image for Streamlit display
        img_with_box_pil = Image.fromarray(img_with_box)

    # Display results
    col1, col2 = st.columns([3, 2])
    
    with col1:
        if prediction[0] == 1:
            st.image(img_with_box_pil, caption='Image with Tumor Area Highlighted', use_column_width=False, width=500)  # Adjust width as needed
        else:
            st.image(uploaded_file, caption='Uploaded Image (No Tumor Detected)', use_column_width=False, width=500)  # Adjust width as needed

    with col2:
        # Display results inside the semi-transparent box
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Prediction:</h2>
            <p><strong>Status:</strong> <em>{prediction_label}</em></p>
            <p><strong>Confidence:</strong> <em>{confidence * 100:.2f}%</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('---')  # Adds a horizontal line
