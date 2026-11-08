import numpy as np
import cv2

##################################
#   Landmark detection utilities
##################################

def get_bbox_from_landmarks(image, landmarks, padding=1.25):
    """
    Calculate a square bounding box from the given landmarks with padding.
    """
    min_x, min_y = np.min(landmarks[:, 0]), np.min(landmarks[:, 1])
    max_x, max_y = np.max(landmarks[:, 0]), np.max(landmarks[:, 1])

    face_width = max_x - min_x
    face_height = max_y - min_y

    face_size = max(face_width, face_height) * padding
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2

    return int(center_x - face_size / 2), int(center_y - face_size / 2), int(face_size)


def crop_face(image, landmarks, padding=1.25):
    """
    Crops the face and shifts landmarks. Handles image boundaries safely.
    """
    x, y, size = get_bbox_from_landmarks(image, landmarks, padding)

    # Calculate safe boundaries
    h, w = image.shape[:2]
    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(w, x + size)
    crop_y2 = min(h, y + size)
    
    # Crop the image (might be rectangular if it hit a boundary)
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Vectorized landmark shift (Much faster than a for loop)
    # We subtract the ACTUAL top-left corner of the crop
    landmarks = landmarks - np.array([crop_x1, crop_y1])

    return cropped_image, landmarks


def resize_image_and_landmarks(image, landmarks, target_size=112):
    """
    Resizes image and correctly scales X and Y landmarks independently.
    """
    h, w = image.shape[:2]
    
    # Calculate independent scaling factors to prevent aspect-ratio distortion bugs
    scale_x = target_size / w
    scale_y = target_size / h
    
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

    # Vectorized scaling
    landmarks = landmarks * np.array([scale_x, scale_y])

    return image, landmarks


def preprocess_face_image_and_landmarks(image: np.ndarray, landmarks: list, target_size=112, padding=1.25):
    """
    Pipeline to crop and resize a face image and its landmarks.
    """
    landmarks = np.array(landmarks, dtype=np.float32)
    
    image, landmarks = crop_face(image, landmarks, padding)
    image, landmarks = resize_image_and_landmarks(image, landmarks, target_size)

    return image, landmarks