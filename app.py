import streamlit as st
import requests
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="wide"
)

# --- Define API URL ---
# This is the address of your FastAPI backend.
# This works because both are running on 'localhost' (your PC).
API_URL = "http://127.0.0.1:8000"

# --- Main App ---
st.title("Deepfake Detector 🕵️")
st.write("Welcome! This app uses two advanced AI models to detect deepfakes in images and videos.")

# --- Create two columns for the detectors ---
col1, col2 = st.columns(2)

# --- Column 1: Image Detector ---
with col1:
    st.header("Deepfake Image Detector")
    image_file = st.file_uploader("Upload an Image", type=['jpg', 'png', 'jpeg'], key="image")

    if image_file:
        # Display the uploaded image
        st.image(image_file, caption="Uploaded Image.", use_container_width=True)
        
        # The 'Detect' button
        if st.button("Detect Image"):
            # Prepare the file for the API
            files = {"file": (image_file.name, image_file, image_file.type)}
            
            # Show a spinner while the API is working
            with st.spinner("Analyzing image... Please wait."):
                try:
                    # Send the request to the FastAPI backend
                    response = requests.post(f"{API_URL}/predict_image", files=files, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        confidence = result.get('confidence', 0.0)
                        
                        if result['prediction'] == 'FAKE':
                            st.error(f"**Prediction: FAKE** (Confidence: {confidence:.2f}%)")
                        else:
                            st.success(f"**Prediction: REAL** (Confidence: {confidence:.2f}%)")
                    else:
                        st.error(f"Error from API: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the backend API: {e}")

# --- Column 2: Video Detector ---
with col2:
    st.header("Deepfake Video Detector")
    video_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'], key="video")

    if video_file:
        # Display the uploaded video
        st.video(video_file)
        
        # The 'Detect' button
        if st.button("Detect Video"):
            # Prepare the file for the API
            files = {"file": (video_file.name, video_file, video_file.type)}
            
            # Show a spinner
            with st.spinner("Analyzing video... This may take a minute or two."):
                try:
                    # Send the request to the FastAPI backend
                    # We give this a much longer timeout
                    response = requests.post(f"{API_URL}/predict_video", files=files, timeout=300)
                    
                    if response.status_code == 200:
                        result = response.json()
                        confidence = result.get('confidence', 0.0)
                        
                        if result['prediction'] == 'FAKE':
                            st.error(f"**Prediction: FAKE** (Confidence: {confidence:.2f}%)")
                        elif result['prediction'] == 'REAL':
                            st.success(f"**Prediction: REAL** (Confidence: {confidence:.2f}%)")
                        else:
                            # Handle the error case from our refactored function
                            st.warning(f"Prediction Error: {result.get('detail', 'Unknown error')}")
                    else:
                        st.error(f"Error from API: {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to the backend API: {e}")