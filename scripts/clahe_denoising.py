import os
import cv2
import argparse
import numpy as np

def adaptive_clahe(image, clip_limit=2.0):
    h, w = image.shape

    # # Determine tile grid size based on image size
    # grid_h = max(2, h // min_tile_size)
    # grid_w = max(2, w // min_tile_size)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    enhanced = clahe.apply(image)
    return enhanced

def process_image(image_path, output_path, clip_limit=2.0):
    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return

  

    # Step 1: Denoise
    denoised = cv2.fastNlMeansDenoising(img, h=10)

    # Step 2: Apply adaptive CLAHE
    enhanced = adaptive_clahe(denoised, clip_limit=clip_limit)

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, enhanced)
    print(f"✅ Saved enhanced image to: {output_path}")

def process_folder(input_dir, output_dir, clip_limit=2.0):
    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(supported_ext):
                in_path = os.path.join(root, fname)
                rel_path = os.path.relpath(in_path, input_dir)
                out_path = os.path.join(output_dir, rel_path)
                process_image(in_path, out_path, clip_limit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise and apply adaptive CLAHE to grayscale images.")
    parser.add_argument("--input_dir", required=True, help="Folder containing input grayscale images")
    parser.add_argument("--output_dir", required=True, help="Folder to save enhanced images")
    parser.add_argument("--clip_limit", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")

    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir, args.clip_limit)

