import os
import cv2
import numpy as np
from google.cloud import vision_v1
from google.cloud import storage
import matplotlib.pyplot as plt


def setup_credentials_img(credentials_path):
    """
    Set up Google Cloud credentials
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    vision_client = vision_v1.ImageAnnotatorClient()
    storage_client = storage.Client()
    return vision_client, storage_client

def detect_objects_and_plates(vision_client, image_path):
    """
    Detect objects and attempt to recognize license plates with interactive visualization
    
    :param vision_client: Google Cloud Vision client
    :param image_path: Path to input image
    :return: Dictionary with detection results
    """
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for matplotlib
    
    # Convert to RGB for GCP Vision
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    vision_image = vision_v1.Image(content=content)
    
    # Object Detection
    object_detection_response = vision_client.object_localization(
        image=vision_image
    )
    objects = object_detection_response.localized_object_annotations
    
    # License Plate Detection
    ocr_response = vision_client.text_detection(image=vision_image)
    texts = ocr_response.text_annotations
    
    # Process detections
    detected_objects = []
    for obj in objects:
        detected_objects.append({
            'name': obj.name,
            'score': obj.score,
            'bounding_poly': obj.bounding_poly
        })
    
    # Extract potential license plate text
    # license_plates = []
    # if texts:
    #     for text in texts[1:]:  # First annotation is full text
    #         license_plates.append(text.description)
    license_plates = [
        text.description for text in texts[1:] 
        if len(text.description.replace(' ', '')) <= 5
    ]
    # Create figure with more space for annotations
    plt.figure(figsize=(15, 10))
    plt.imshow(image_rgb)
    plt.title('Object and License Plate Detection')
    plt.axis('off')  # Hide axis
    
    # Draw bounding boxes for objects
    for obj in detected_objects:
        pts = [(p.x, p.y) for p in obj['bounding_poly'].normalized_vertices]
        pts = np.array([(p[0]*image.shape[1], p[1]*image.shape[0]) for p in pts])
        
        # Plot bounding box
        plt.plot([pts[0][0], pts[1][0], pts[2][0], pts[3][0], pts[0][0]], 
                 [pts[0][1], pts[1][1], pts[2][1], pts[3][1], pts[0][1]], 
                 color='red', linewidth=2)
        
        # Add label with confidence
        plt.text(pts[0][0], pts[0][1], 
                 f"{obj['name']} ({obj['score']:.2f})", 
                 color='white', 
                 bbox=dict(facecolor='red', alpha=0.7))
    
    # Show the plot
    # plt.show()
    
    return {
        'objects': detected_objects,
        'license_plates': license_plates
    }
