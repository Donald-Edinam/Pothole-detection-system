from ultralytics import YOLO
import cv2
import numpy as np
import json

# Load the YOLOv8 model
model = YOLO('yolov8n-seg.pt')  # or your custom trained model

# Function to calculate area of a contour
def calculate_area(contour):
    return cv2.contourArea(contour)

# Process images and store results
results = []
id_counter = 1

# Assuming you have a list of image paths
image_paths = ['/home/lusitech/ComputerVision/PotholeDetection/Potholes.v1i.coco/train/p6_jpeg.rf.f72f8c7d6de01d953d4b317cb724c0a2.jpg']  # Add your image paths here

for img_path in image_paths:
    # Perform inference
    result = model(img_path, task='segment')[0]
    
    # Get the original image dimensions
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # Process each detected object
    for seg in result.masks.segments:
        # Convert normalized coordinates to pixel coordinates
        contour = np.array(seg * [width, height], dtype=np.int32)
        
        # Calculate area
        area = calculate_area(contour)
        
        # Store result
        results.append({
            'id': id_counter,
            'image': img_path,
            'area': float(area)  # Convert to float for JSON serialization
        })
        
        id_counter += 1

# Save results to JSON file
with open('pothole_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to pothole_results.json")