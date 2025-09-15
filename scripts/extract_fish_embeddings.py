import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
import timm
from collections import OrderedDict
from tqdm import tqdm

# === Settings ===
weights_path = "..models_and_weights/resnet50_Family_FishNet.pt"
model_name = "resnet50"  # Change if you're using fishnet99, fishnet150, etc.

# === Device ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# === Load Model ===
def load_fishnet_model():
    model = timm.create_model(model_name, pretrained=False, num_classes=1000)
    checkpoint = torch.load(weights_path, map_location=device)

    # Handle state dict keys
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    new_state = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            k = k[6:]
        new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    model.to(device)
    return model

model = load_fishnet_model()

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === Feature Extraction ===
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(image)
        if isinstance(features, (tuple, list)):
            features = features[0]
    return features.squeeze().cpu().numpy()

# === Directory Walker ===
def extract_embeddings_from_directory(directory_path):
    embeddings = []
    image_paths = []

    for filename in tqdm(sorted(os.listdir(directory_path))):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory_path, filename)
            try:
                feat = extract_features(image_path)
                embeddings.append(feat)
                image_paths.append(image_path)
            except Exception as e:
                print(f"⚠️ Skipped {filename}: {e}")

    embeddings = np.stack(embeddings)
    return embeddings, image_paths

# === Save Embeddings ===
def save_to_npz(embeddings, image_paths, output_file):
    np.savez_compressed(output_file, embeddings=embeddings, image_paths=image_paths)
    print(f"✅ Saved {len(image_paths)} embeddings to {output_file}")

# === Main CLI ===
def main():
    parser = argparse.ArgumentParser(description="Extract compressed FishNet embeddings.")
    parser.add_argument('--image_dir', type=str, required=True, help="Folder with images.")
    parser.add_argument('--output_file', type=str, required=True, help="Output .npz file.")
    args = parser.parse_args()

    embeddings, image_paths = extract_embeddings_from_directory(args.image_dir)
    save_to_npz(embeddings, image_paths, args.output_file)

if __name__ == "__main__":
    main()
