import os
import re
import argparse

def rename_files(folder_path, new_class_id):
    # Validate class ID format
    if not new_class_id.isdigit():
        print("❌ Class ID must be numeric.")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            match = re.match(r'^(.*)_\d+(\.jpg)$', filename)
            if match:
                new_filename = f"{match.group(1)}_{new_class_id}{match.group(2)}"
                src = os.path.join(folder_path, filename)
                dst = os.path.join(folder_path, new_filename)
                os.rename(src, dst)
                print(f"🔁 Renamed: {filename} → {new_filename}")
            else:
                print(f"⚠️ Skipped (no class ID found): {filename}")

def main():
    parser = argparse.ArgumentParser(description="Rename image filenames to change class ID.")
    parser.add_argument('--folder', type=str, required=True, help='Path to the folder with images')
    parser.add_argument('--new_class', type=str, required=True, help='New class ID to replace the current one')

    args = parser.parse_args()
    rename_files(args.folder, args.new_class)

if __name__ == "__main__":
    main()

