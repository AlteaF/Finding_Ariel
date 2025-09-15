
import os

import numpy as np
import cv2 as cv



def convert_to_clahe(image_path, output_path):
    """Converts an image to clahe and saves it."""
    try:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        cv.imwrite(output_path, cl1)
        print(f"Successfully converted {image_path} to {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_dataset(input_folder, output_folder):
    """Processes all images in a folder to clahe."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            convert_to_clahe(image_path, output_path)

# Example Usage
input_dir = "/Users/alteafogh/Documents/ITU/summer/data/train/images" # Replace with your input directory
output_dir = "/Users/alteafogh/Documents/ITU/summer/data/train/clahe" # Replace with your desired output directory
process_dataset(input_dir, output_dir)
