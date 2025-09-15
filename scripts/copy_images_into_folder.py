import os
import shutil
import re
import argparse

def extract_class_id(filename):
    """Extract the class number (last digits before the .jpg extension, after an underscore)."""
    match = re.findall(r'_(\d+)\.jpg$', filename)
    return match[0] if match else None

def organize_images(source_dir, dest_dir, selected_classes=None):
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".jpg"):
            class_id = extract_class_id(filename)

            if class_id is not None:
                if selected_classes and class_id not in selected_classes:
                    continue  # Skip if class filtering is active and this class isn't selected

                class_folder = os.path.join(dest_dir, class_id)
                os.makedirs(class_folder, exist_ok=True)

                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(class_folder, filename)

                shutil.copy2(src_path, dst_path)
                print(f"✅ Copied {filename} → {class_folder}")
            else:
                print(f"⚠️ Could not extract class from: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Copy images to subfolders based on class ID.")
    parser.add_argument('--source', type=str, required=True, help='Path to folder with original .jpg images')
    parser.add_argument('--destination', type=str, required=True, help='Path to output folder to copy organized images')
    parser.add_argument('--classes', nargs='*', help='List of class IDs to include (e.g. 0 1 2). If omitted, includes all.')

    args = parser.parse_args()

    selected_classes = set(args.classes) if args.classes else None
    organize_images(args.source, args.destination, selected_classes)

if __name__ == "__main__":
    main()
