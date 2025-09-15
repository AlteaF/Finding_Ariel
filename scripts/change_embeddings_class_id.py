import argparse
import numpy as np
import os
import re
import csv

def parse_mapping(map_list):
    mapping = {}
    for item in map_list:
        old, new = item.split(':')
        mapping[int(old)] = int(new)
    return mapping

def remap_class_in_path(image_path, class_map):
    filename = os.path.basename(image_path)
    dirname = os.path.dirname(image_path)
    match = re.findall(r'(\d+)(?=\.jpg$)', filename)
    
    if not match:
        return image_path, 'error: no class ID'
    
    old_class_id = int(match[-1])
    
    if old_class_id not in class_map:
        return image_path, 'unchanged'
    
    new_class_id = class_map[old_class_id]
    new_filename = re.sub(r'(\d+)(?=\.jpg$)', str(new_class_id), filename, count=1)
    new_path = os.path.join(dirname, new_filename)
    
    return new_path, 'remapped'

def main():
    parser = argparse.ArgumentParser(description="Remap class IDs in image paths within a .npz file, with CSV summary.")
    parser.add_argument('--input', type=str, required=True, help='Path to input .npz file with embeddings and image_paths')
    parser.add_argument('--output', type=str, required=True, help='Path to save output .npz file with updated image paths')
    parser.add_argument('--map', nargs='+', required=True, help='List of class ID remaps in the form old:new (e.g. 1:0 3:2)')
    parser.add_argument('--log_csv', type=str, default='remap_log.csv', help='CSV file to log mapping results')
    args = parser.parse_args()

    print("🔍 Loading input file...")
    data = np.load(args.input, allow_pickle=True)
    embeddings = data['embeddings']
    image_paths = data['image_paths']

    class_map = parse_mapping(args.map)
    print(f"🔁 Applying class ID mapping: {class_map}")

    new_image_paths = []
    log_entries = []

    for original_path in image_paths:
        new_path, status = remap_class_in_path(original_path, class_map)
        new_image_paths.append(new_path)
        log_entries.append({'original_path': original_path, 'new_path': new_path, 'status': status})

    print(f"💾 Saving updated .npz file to {args.output}...")
    np.savez(args.output, embeddings=embeddings, image_paths=new_image_paths)

    print(f"📝 Writing log to {args.log_csv}...")
    with open(args.log_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['original_path', 'new_path', 'status'])
        writer.writeheader()
        writer.writerows(log_entries)

    print("✅ Done. You can inspect the log file to verify changes.")

if __name__ == '__main__':
    main()
