from flask import Flask, request, jsonify , send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
sys.path.append('.')  # Add current directory to path

from image import setup_credentials_img, detect_objects_and_plates
from video import setup_credentials, process_video, aggregate_object_detections
from google.cloud import vision_v1

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload directory
UPLOAD_FOLDER = 'uploads'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set up Google Cloud credentials path
CREDENTIALS_PATH = 'hqanpr-d27230c75bb2.json'

def allowed_file(filename, file_type='image'):
    """Check if file has an allowed extension"""
    if file_type == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

@app.route('/process_images', methods=['POST'])
def process_images():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    results = []
    
    vision_client, _ = setup_credentials_img(CREDENTIALS_PATH)
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process image
                detection_results = detect_objects_and_plates(vision_client, filepath)
                
                # Prepare results with some processing
                results.append({
                    'filename': filename,
                    'license_plates': detection_results['license_plates'],
                    'objects': list(set(obj['name'] for obj in detection_results['objects']))
                })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'error': str(e)
                })
    
    return jsonify(results)

@app.route('/process_videos', methods=['POST'])
def process_videos():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    results = []
    
    vision_client = setup_credentials(CREDENTIALS_PATH)
    
    for file in files:
        if file and allowed_file(file.filename, 'video'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Process video
                video_results = process_video(vision_client, filepath)
                
                # Aggregate object detections
                unique_objects = aggregate_object_detections(video_results['all_objects'])
                
                results.append({
                    'filename': filename,
                    'license_plates': list(set(video_results['all_license_plates']))[:5],  # First 5 unique plates
                    'objects': [f"{name} (Count: {info['count']})" for name, info in unique_objects.items()]
                })
            except Exception as e:
                results.append({
                    'filename': filename,
                    'error': str(e)
                })
    
    return jsonify(results)

@app.route('/')
def serve_html():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    app.run(host='192.168.1.5',debug=True)
