import os
import cv2
import numpy as np
from google.cloud import vision_v1
from google.cloud import storage
import matplotlib.pyplot as plt


def setup_credentials(credentials_path):
    """
    Set up Google Cloud credentials
    :param credentials_path: Path to Google Cloud credentials JSON
    :return: Vision client
    """
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    return vision_v1.ImageAnnotatorClient()

def process_video(vision_client, video_path, frame_interval=5):
    """
    Process video frames for object and license plate detection
    :param vision_client: Google Cloud Vision client
    :param video_path: Path to input video
    :param frame_interval: Process every nth frame
    :return: Comprehensive detection results
    """
    video = cv2.VideoCapture(video_path)
    all_license_plates = []
    all_objects = []
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_interval == 0:
            # Convert frame to bytes for Vision API
            _, buffer = cv2.imencode('.jpg', frame)
            content = buffer.tobytes()
            
            vision_image = vision_v1.Image(content=content)
            
            # Object Detection
            object_detection_response = vision_client.object_localization(
                image=vision_image
            )
            objects = object_detection_response.localized_object_annotations
            
            # License Plate Detection
            ocr_response = vision_client.text_detection(image=vision_image)
            texts = ocr_response.text_annotations
            
            # Process objects
            frame_objects = []
            for obj in objects:
                obj_info = {
                    'name': obj.name,
                    'score': obj.score,
                    'bounding_poly': obj.bounding_poly
                }
                frame_objects.append(obj_info)
                all_objects.append(obj_info)
            
            # Extract license plates
            if texts:
                for text in texts[1:]:  # First annotation is full text
                    if text.description not in all_license_plates:
                        all_license_plates.append(text.description)
            
            # Visualize in real-time
            vis_frame = frame.copy()
            for obj in frame_objects:
                pts = [(p.x, p.y) for p in obj['bounding_poly'].normalized_vertices]
                pts = np.array([(int(p[0]*frame.shape[1]), int(p[1]*frame.shape[0])) for p in pts])
                
                # Draw bounding box
                cv2.polylines(vis_frame, [pts], True, (0, 255, 0), 2)
                
                # Add label with confidence
                label = f"{obj['name']} ({obj['score']:.2f})"
                cv2.putText(vis_frame, label, 
                            (int(pts[0][0]), int(pts[0][1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (0, 255, 0), 2)
            
            # Show frame
            # cv2.imshow('Video Detection', vis_frame)
            
            # Wait for key press (30ms delay)
            key = cv2.waitKey(30)
            
            # Break loop if 'q' is pressed
            if key == ord('q'):
                break
    
    # Close video window
    cv2.destroyAllWindows()
    
    return {
        'all_license_plates': all_license_plates,
        'all_objects': all_objects
    }

def aggregate_object_detections(all_objects):
    """
    Aggregate object detections to get unique object counts and max confidences
    :param all_objects: List of detected objects
    :return: Dictionary of aggregated object information
    """
    unique_objects = {}
    for obj in all_objects:
        if obj['name'] not in unique_objects:
            unique_objects[obj['name']] = {
                'count': 1,
                'max_confidence': obj['score']
            }
        else:
            unique_objects[obj['name']]['count'] += 1
            unique_objects[obj['name']]['max_confidence'] = max(
                unique_objects[obj['name']]['max_confidence'], 
                obj['score']
            )
    return unique_objects
