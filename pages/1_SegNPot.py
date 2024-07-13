import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st
import json
import folium
from streamlit_folium import folium_static
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from ultralytics import YOLO
from io import BytesIO

from sample_utils.download import download_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Road Damage Segmentation",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

MODEL_URL = "https://github.com/hamdani2020/Pothole/raw/main/models/best.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Seg.pt"

# Ensure the models directory exists
os.makedirs(ROOT / "models", exist_ok=True)

def download_file_with_logging(url, local_path, expected_size):
    logger.info(f"Downloading file from {url} to {local_path}")
    download_file(url, local_path, expected_size)
    logger.info(f"Download complete. File saved to {local_path}")

@st.cache_data(persist="disk")
def load_model():
    logger.info("Entering load_model function")
    if not MODEL_LOCAL_PATH.exists():
        logger.info(f"Model not found at {MODEL_LOCAL_PATH}. Downloading...")
        download_file_with_logging(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)
    else:
        logger.info(f"Model found at {MODEL_LOCAL_PATH}")
    logger.info("Loading YOLO model")
    model = YOLO(MODEL_LOCAL_PATH)
    logger.info("YOLO model loaded successfully")
    return model

# Load the model
logger.info("About to load the model")
net = load_model()
logger.info("Model loading complete")

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Segmentation(NamedTuple):
    class_id: int
    label: str
    score: float
    mask: np.ndarray
    area: float

def get_decimal_coordinates(info):
    for key in ['Latitude', 'Longitude']:
        if key not in info:
            return None, None

    lat = info['Latitude']
    lon = info['Longitude']
    lat_ref = info['GPSLatitudeRef']
    lon_ref = info['GPSLongitudeRef']

    lat = lat[0] + lat[1] / 60 + lat[2] / 3600
    lon = lon[0] + lon[1] / 60 + lon[2] / 3600

    if lat_ref != 'N':
        lat = -lat
    if lon_ref != 'E':
        lon = -lon

    return lat, lon

def extract_gps_info(image_file):
    try:
        image = Image.open(image_file)
        exif_data = {}
        info = image._getexif()
        if info:
            for tag_id, value in info.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == "GPSInfo":
                    gps_data = {}
                    for gps_tag_id in value:
                        gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                        gps_data[gps_tag] = value[gps_tag_id]
                    exif_data[tag] = gps_data
                else:
                    exif_data[tag] = value

        if "GPSInfo" in exif_data:
            gps_info = exif_data["GPSInfo"]
            lat, lon = get_decimal_coordinates(gps_info)
            return lat, lon
        else:
            return None, None
    except Exception as e:
        logger.error(f"Error extracting GPS info: {str(e)}")
        return None, None

st.title("Pothole Segmentation - Multiple Images with GPS")
st.write("Segment road damage using multiple images. Upload a folder of images and start segmenting. This section can be useful for processing batch data.")

uploaded_files = st.file_uploader("Upload Images", type=['png', 'jpg'], accept_multiple_files=True)

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction.")

def calculate_area(mask):
    return np.sum(mask)

def process_image(image_file):
    logger.info(f"Processing image: {image_file.name}")
    image = Image.open(image_file)
    _image = np.array(image)
    h_ori, w_ori = _image.shape[:2]

    image_resized = cv2.resize(_image, (720, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    segmentations = []
    for result in results:
        masks = result.masks
        boxes = result.boxes
        if masks is not None:
            for i, (mask, box) in enumerate(zip(masks.data, boxes)):
                mask = mask.cpu().numpy()
                mask = cv2.resize(mask, (w_ori, h_ori))
                area = calculate_area(mask)
                segmentations.append(
                    Segmentation(
                        class_id=int(box.cls),
                        label=CLASSES[int(box.cls)],
                        score=float(box.conf),
                        mask=mask,
                        area=float(area)
                    )
                )

    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    # Extract GPS coordinates
    lat, lon = extract_gps_info(image_file)

    return _image, _image_pred, segmentations, lat, lon

if uploaded_files:
    logger.info(f"{len(uploaded_files)} images uploaded. Processing...")
    
    # Create a map centered on the first image with GPS data
    m = folium.Map(location=[0, 0], zoom_start=2)
    
    all_results = []
    
    for image_file in uploaded_files:
        st.write(f"### Processing: {image_file.name}")
        
        original_image, predicted_image, segmentations, lat, lon = process_image(image_file)
        
        col1, col2 = st.columns(2)
        
        # Original Image
        with col1:
            st.write("#### Original Image")
            st.image(original_image)
        
        # Predicted Image
        with col2:
            st.write("#### Predictions")
            st.image(predicted_image)

            # Download predicted image
            buffer = BytesIO()
            _downloadImages = Image.fromarray(predicted_image)
            _downloadImages.save(buffer, format="PNG")
            _downloadImagesByte = buffer.getvalue()

            st.download_button(
                label="Download Prediction Image",
                data=_downloadImagesByte,
                file_name=f"Predicted_{image_file.name}",
                mime="image/png"
            )

        # Display segmentation results
        st.write("#### Segmentation Results")
        for seg in segmentations:
            st.write(f"Class: {seg.label}, Confidence: {seg.score:.2f}, Area: {seg.area:.2f} pixels")

        # Save results as JSON
        json_results = {
            "filename": image_file.name,
            "gps": {"latitude": lat, "longitude": lon},
            "segmentations": [
                {
                    "id": i,
                    "class": seg.label,
                    "confidence": float(seg.score),
                    "area": float(seg.area)
                } for i, seg in enumerate(segmentations)
            ]
        }
        
        all_results.append(json_results)

        json_str = json.dumps(json_results, indent=2)
        st.download_button(
            label="Download Results as JSON",
            data=json_str,
            file_name=f"segmentation_results_{image_file.name}.json",
            mime="application/json"
        )
        
        # Add marker to the map if GPS data is available
        if lat is not None and lon is not None:
            folium.Marker(
                [lat, lon], 
                popup=f"Image: {image_file.name}<br>Potholes: {sum(1 for seg in segmentations if seg.label == 'Potholes')}"
            ).add_to(m)
        
        st.write("---")  # Add a separator between images

    # Display the map
    st.write("### Map of Processed Images")
    folium_static(m)

    # Offer download of all results
    all_results_json = json.dumps(all_results, indent=2)
    st.download_button(
        label="Download All Results as JSON",
        data=all_results_json,
        file_name="all_segmentation_results.json",
        mime="application/json"
    )

    logger.info("All images processed")

logger.info("Streamlit app execution complete")