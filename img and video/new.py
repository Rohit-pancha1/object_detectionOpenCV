# import cv2
# import numpy as np

# def find_image_in_video(video_path, image_path, save_dir):
#     # Read the query image
#     query_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     if query_img is None:
#         print(f"Error: Could not read image {image_path}")
#         return
    
#     # Convert query image to grayscale
#     if query_img.ndim == 3:
#         query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
#     else:
#         query_gray = query_img

#     # Read the video
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open video.")
#         return
    
#     # Get dimensions of the query image
#     h, w = query_gray.shape[:2]
    
#     # Create a method for comparison
#     method = cv2.TM_CCOEFF_NORMED
    
#     frame_number = 0
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Convert frame to grayscale
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Perform template matching
#         result = cv2.matchTemplate(frame_gray, query_gray, method)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
#         # Specify a threshold to consider it a match
#         threshold = 0.8
#         if max_val >= threshold:
#             # Draw rectangle around the matched region
#             top_left = max_loc
#             bottom_right = (top_left[0] + w, top_left[1] + h)
#             cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            
#             # Save the frame as an image
#             save_path = f"{save_dir}/frame_{frame_number}.jpg"
#             cv2.imwrite(save_path, frame)
#             print(f"Found image in frame {frame_number}. Saved as {save_path}")
        
#         frame_number += 1
    
#     # Release the capture object and close any open windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Example usage
# if __name__ == "__main__":
#     video_path = 'one.mp4'
#     image_path = 'image1.png', 'image.png'
#     save_directory = 'output_frames'
    
#     # Ensure the save directory exists
#     import os
#     os.makedirs(save_directory, exist_ok=True)
    
#     find_image_in_video(video_path, image_path, save_directory)


import cv2
import os

def find_image_in_video(video_path, image_paths, save_dir):
    # Read each query image and convert to grayscale
    query_images = []
    for image_path in image_paths:
        query_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if query_img is None:
            print(f"Error: Could not read image {image_path}")
            continue
        
        if query_img.ndim == 3:
            query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        else:
            query_gray = query_img
        
        query_images.append(query_gray)

    if not query_images:
        print("No valid query images found.")
        return
    
    # Read the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get dimensions of the first query image
    h, w = query_images[0].shape[:2]
    
    # Create a method for comparison
    method = cv2.TM_CCOEFF_NORMED
    
    frame_number = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        found_match = False
        for i, query_gray in enumerate(query_images):
            # Perform template matching
            result = cv2.matchTemplate(frame_gray, query_gray, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Print maximum correlation value for debugging
            print(f"Max correlation value for image {i}: {max_val}")
            
            # Specify a threshold to consider it a match
            threshold = 0.7  # Adjust this value as needed
            if max_val >= threshold:
                # Draw rectangle around the matched region
                top_left = max_loc
                bottom_right = (top_left[0] + w, top_left[1] + h)
                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                
                # Save the frame as an image
                save_path = os.path.join(save_dir, f"frame_{frame_number}_match_{i}.jpg")
                cv2.imwrite(save_path, frame)
                print(f"Found image {i} in frame {frame_number}. Saved as {save_path}")
                
                found_match = True
        
        if found_match:
            # Break out of the loop if any image was found in the frame
            break
        
        frame_number += 1
    
    # Release the capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = 'one.mp4'
    image_paths = ['image1.png'] # List of image paths
    save_directory = 'output_frames'
    
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)
    
    find_image_in_video(video_path, image_paths, save_directory)
