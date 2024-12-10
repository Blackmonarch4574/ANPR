import os
from image import *
from video import *

def main():
    # Set up credentials path
    credentials_path = 'f:/comand HQ/hqanpr-d27230c75bb2.json'
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    input_folder = 'f:/comand HQ/project/input'

    while True:
        # Ask user for input type: 1 for image, 2 for video, or 'exit' to quit
        input_type = input("Enter 1 for image processing, 2 for video processing, or 'exit' to quit: ").strip().lower()

        if input_type == '1':
            # Image Processing

            vision_client, _ = setup_credentials_img(credentials_path)
    
            # Input folder path
            input_folder = 'f:/comand HQ/project/input'

            # Process images from input folder
            for filename in os.listdir(input_folder):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(input_folder, filename)
                    print(f"\nProcessing image: {image_path}")
                    
                    # Detect objects and plates
                    results = detect_objects_and_plates(vision_client, image_path)
                    
                    # Print License Plates
                    print("\n📍 License Plates Found:")
                    if results['license_plates']:
                        for plate in results['license_plates']:
                            print(f"   - {plate}")
                    else:
                        print("   No license plates detected.")

                    
                    
                    # Print Detected Objects
                    # print("\n🚗 Detected Objects:")
                    # for obj in results['objects']:
                    #     print(f"   - {obj['name']} (Confidence: {obj['score']:.2f})")
                    print("\n🚗 Detected Objects:")
                    unique_objects = set(obj['name'] for obj in results['objects'])
                    for obj in unique_objects:
                        print(f"   - {obj}")

        elif input_type == '2':
            # Video Processing
            vision_client = setup_credentials(credentials_path)
    
            # Process videos from input folder
            for filename in os.listdir(input_folder):
                if filename.endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(input_folder, filename)
                    print(f"\nProcessing video: {video_path}")
                    
                    # Process video frames
                    results = process_video(vision_client, video_path)
                    
                    # Aggregate object detections
                    unique_objects = aggregate_object_detections(results['all_objects'])
                    
                    # Print Final Results
                    print("\n🚗 All Detected Objects:")
                    objects_list = [
                        f"{name} (Count: {info['count']})" 
                        for name, info in unique_objects.items()
                    ]
                    print("[\n    " + ",\n    ".join(objects_list) + "\n]")
                    
                    # Print Final License Plates
                    # print("\n📍 All License Plates:")
                    # print("[\n    " + ",\n    ".join(results['all_license_plates']) + "\n]")
                    print("\n📍 License Plates:")
                    sorted_plates = sorted(set(results['all_license_plates']))
                    for plate in sorted_plates[:5]:  # Limit to 5 plates
                        print(f"   - {plate}")

        elif input_type == 'exit':
            print("Exiting the program.")
            break

        else:
            print("Invalid input. Please enter 1 for image processing, 2 for video processing, or 'exit' to quit.")

if __name__ == '__main__':
    main()
